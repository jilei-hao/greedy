/*=========================================================================

  Program:   ALFABIS fast medical image registration programs
  Language:  C++
  Website:   github.com/pyushkevich/greedy
  Copyright (c) Paul Yushkevich, University of Pennsylvania. All rights reserved.

  This program is part of ALFABIS: Adaptive Large-Scale Framework for
  Automatic Biomedical Image Segmentation.

  ALFABIS development is funded by the NIH grant R01 EB017255.

  ALFABIS is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  ALFABIS is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ALFABIS.  If not, see <http://www.gnu.org/licenses/>.

=========================================================================*/
#include "GreedyAPI.h"

#include <iostream>
#include <sstream>
#include <cstdio>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdarg>

#include "lddmm_common.h"
#include "lddmm_data.h"

#include <itkImageFileReader.h>
#include <itkAffineTransform.h>
#include <itkTransformFactory.h>
#include <itkTimeProbe.h>
#include <itkImageFileWriter.h>
#include <itkCastImageFilter.h>
#include <itkDiscreteGaussianImageFilter.h>
#include <itkResampleImageFilter.h>
#include <itkIdentityTransform.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkBinaryBallStructuringElement.h>
#include <itkBinaryDilateImageFilter.h>

#include "ImageRegionConstIteratorWithIndexOverride.h"

#include "MultiImageRegistrationHelper.h"
#include "FastWarpCompositeImageFilter.h"
#include "MultiComponentImageMetricBase.h"
#include "WarpFunctors.h"

#include <vtkPolyData.h>
#include "GreedyMeshIO.h"

#include <vnl/algo/vnl_powell.h>
#include <vnl/algo/vnl_amoeba.h>
#include <vnl/algo/vnl_svd.h>
#include <vnl/algo/vnl_symmetric_eigensystem.h>
#include <vnl/vnl_trace.h>
#include <vnl/vnl_numeric_traits.h>
#include <vnl/algo/vnl_lbfgs.h>
#include <vnl/algo/vnl_conjugate_gradient.h>


// A helper class for printing results. Wraps around printf but
// takes into account the user's verbose settings
class GreedyStdOut
{
public:
  GreedyStdOut(GreedyParameters::Verbosity verbosity, FILE *f_out = NULL)
  : m_Verbosity(verbosity), m_Output(f_out ? f_out : stdout)
  {
  }

  void printf(const char *format, ...)
  {
    if(m_Verbosity > GreedyParameters::VERB_NONE)
      {
      char buffer[4096];
      va_list args;
      va_start (args, format);
      vsprintf (buffer,format, args);
      va_end (args);

      fprintf(m_Output, "%s", buffer);
      }
  }

  void flush()
  {
    fflush(m_Output);
  }

private:
  GreedyParameters::Verbosity m_Verbosity;
  FILE *m_Output;

};


// Helper function to get the RAS coordinate of the center of
// an image
template <unsigned int VDim>
vnl_vector<double>
GetImageCenterinNiftiSpace(itk::ImageBase<VDim> *image)
{
  itk::ImageRegion<VDim> r = image->GetBufferedRegion();
  itk::ContinuousIndex<double, VDim> idx;
  itk::Point<double, VDim> ctr;
  for(unsigned int d = 0; d < VDim; d++)
    idx[d] = r.GetIndex()[d] + r.GetSize()[d] * 0.5;
  image->TransformContinuousIndexToPhysicalPoint(idx, ctr);

  // Map to RAS (flip first two coordinates)
  for(unsigned int d = 0; d < 2 && d < VDim; d++)
    ctr[d] = -ctr[d];

  return ctr.GetVnlVector();
}




#include "itkTransformFileReader.h"

template <unsigned int VDim, typename TReal>
vnl_matrix<double>
GreedyApproach<VDim, TReal>
::ReadAffineMatrixViaCache(const TransformSpec &ts)
{
  // Physical (RAS) space transform matrix
  vnl_matrix<double> Qp(VDim+1, VDim+1);  Qp.set_identity();

  // An ITK-style transform - forced to floating point here
  typedef itk::MatrixOffsetTransformBase<double, VDim, VDim> TransformType;
  typename TransformType::Pointer itk_tran;

  // See if a transform is already stored in the cache
  typename ImageCache::const_iterator itCache = m_ImageCache.find(ts.filename);
  if(itCache != m_ImageCache.end())
    {
    TransformType *cached = dynamic_cast<TransformType *>(itCache->second.target);
    if(!cached)
      throw GreedyException("Cached transform %s cannot be cast to type %s",
                            ts.filename.c_str(), typeid(TransformType).name());

    itk_tran = cached;
    }
  else
    {
    // Open the file and read the first line
    std::ifstream fin(ts.filename.c_str());
    std::string header_line, itk_header = "#Insight Transform File";
    std::getline(fin, header_line);

    if(header_line.substr(0, itk_header.size()) == itk_header)
      {
      fin.close();
      try
        {
        // First we try to load the transform using ITK code
        // This code is from c3d_affine_tool
        typedef itk::AffineTransform<double, VDim> AffTran;
        itk::TransformFactory<TransformType>::RegisterTransform();
        itk::TransformFactory<AffTran>::RegisterTransform();

        itk::TransformFileReader::Pointer fltReader = itk::TransformFileReader::New();
        fltReader->SetFileName(ts.filename.c_str());
        fltReader->Update();

        itk::TransformBase *base = fltReader->GetTransformList()->front();
        itk_tran = dynamic_cast<TransformType *>(base);
        }
      catch(...)
        {
        throw GreedyException("Unable to read ITK transform file %s", ts.filename.c_str());
        }
      }
    else
      {
      // Try reading C3D matrix format
      fin.seekg(0);
      for(size_t i = 0; i < VDim+1; i++)
        for(size_t j = 0; j < VDim+1; j++)
          if(fin.good())
            {
            fin >> Qp[i][j];
            }
      fin.close();
      }
    }

  // At this point we might have read the RAS matrix directly, or an ITK transform
  // if the latter, extract the RAS matrix
  if(itk_tran.IsNotNull())
    {
    for(size_t r = 0; r < VDim; r++)
      {
      for(size_t c = 0; c < VDim; c++)
        {
        Qp(r,c) = itk_tran->GetMatrix()(r,c);
        }
      Qp(r,VDim) = itk_tran->GetOffset()[r];
      }

    // RAS - LPI nonsense
    if(VDim == 3)
      {
      Qp(2,0) *= -1; Qp(2,1) *= -1;
      Qp(0,2) *= -1; Qp(1,2) *= -1;
      Qp(0,3) *= -1; Qp(1,3) *= -1;
      }
    }

  // Compute the exponent and its logarithm
  double abs_exponent = fabs(ts.exponent);
  int log_abs_exponent = int(log2(abs_exponent) + 0.5);
  if(abs_exponent != (int) (pow(2.0, log_abs_exponent) + 0.5))
    throw GreedyException("Transform exponent must be a power of 2");

  // Compute the exponent
  if(ts.exponent == 1.0)
    {
    }
  else if(ts.exponent == -1.0)
    {
    Qp = vnl_matrix_inverse<double>(Qp).as_matrix();
    }
  else if(ts.exponent > 0)
    {
    // Multiply the matrix by itself
    for(int j = 0; j < log_abs_exponent; j++)
      Qp = Qp * Qp;
    }
  else if(ts.exponent < 0)
    {
    // Compute the matrix square root
    for(int j = 0; j < log_abs_exponent; j++)
      {
      // Peform Denman-Beavers iteration
      typedef vnl_matrix_fixed<double, VDim+1, VDim+1> MatrixType;
      MatrixType Z, Y = Qp;
      Z.set_identity();

      for(size_t i = 0; i < 16; i++)
        {
        MatrixType Ynext = 0.5 * (Y + vnl_matrix_inverse<double>(Z.as_matrix()).as_matrix());
        MatrixType Znext = 0.5 * (Z + vnl_matrix_inverse<double>(Y.as_matrix()).as_matrix());
        Y = Ynext;
        Z = Znext;
        }

      Qp = Y.as_matrix();
      }
    }

  return Qp;
}




template <unsigned int VDim, typename TReal>
void
GreedyApproach<VDim, TReal>
::WriteAffineMatrixViaCache(
    const std::string &filename, const vnl_matrix<double> &Qp)
{
  // An ITK-style transform - forced to double point here
  typedef itk::MatrixOffsetTransformBase<double, VDim, VDim> TransformType;

  // See if a transform is already stored in the cache
  typename ImageCache::const_iterator itCache = m_ImageCache.find(filename);
  if(itCache != m_ImageCache.end())
    {
    TransformType *cached = dynamic_cast<TransformType *>(itCache->second.target);
    if(!cached)
      throw GreedyException("Cached transform %s cannot be cast to type %s",
                            filename.c_str(), typeid(TransformType).name());

    // RAS - LPI nonsense
    vnl_matrix<double> Q = Qp;
    if(VDim == 3)
      {
      Q(2,0) *= -1; Q(2,1) *= -1;
      Q(0,2) *= -1; Q(1,2) *= -1;
      Q(0,3) *= -1; Q(1,3) *= -1;
      }

    typename TransformType::MatrixType matrix;
    typename TransformType::OffsetType offset;

    // We have found the output transform and can use it for assignment
    for(size_t r = 0; r < VDim; r++)
      {
      for(size_t c = 0; c < VDim; c++)
        {
        matrix(r, c) = Q(r, c);
        }
      offset[r] = Q(r, VDim);
      }

    cached->SetMatrix(matrix);
    cached->SetOffset(offset);
    }

  // Write to actual file
  if(itCache == m_ImageCache.end() || itCache->second.force_write)
    {
    std::ofstream matrixFile;
    matrixFile.open(filename.c_str());
    matrixFile << Qp;
    matrixFile.close();
    }
}

template<unsigned int VDim, typename TReal>
vnl_matrix<double>
GreedyApproach<VDim, TReal>
::ReadAffineMatrix(const TransformSpec &ts)
{
  GreedyApproach<VDim, TReal> api;
  return api.ReadAffineMatrixViaCache(ts);
}

template<unsigned int VDim, typename TReal>
void
GreedyApproach<VDim, TReal>
::ReadAffineTransform(const TransformSpec &ts, LinearTransformType *out_tran)
{
  vnl_matrix<double> Q = ReadAffineMatrix(ts);
  vnl_matrix<double>  A = Q.extract(VDim, VDim);
  vnl_vector<double> b = Q.get_column(VDim).extract(VDim);

  typename LinearTransformType::MatrixType tran_A;
  typename LinearTransformType::OffsetType tran_b;

  vnl_matrix_to_itk_matrix(A, tran_A);
  vnl_vector_to_itk_vector(b, tran_b);

  out_tran->SetMatrix(tran_A);
  out_tran->SetOffset(tran_b);
}



template<unsigned int VDim, typename TReal>
void
GreedyApproach<VDim, TReal>
::WriteAffineMatrix(const std::string &filename, const vnl_matrix<double> &Qp)
{
  GreedyApproach<VDim, TReal> api;
  api.WriteAffineMatrixViaCache(filename, Qp);
}

template<unsigned int VDim, typename TReal>
void
GreedyApproach<VDim, TReal>
::WriteAffineTransform(const std::string &filename, LinearTransformType *tran)
{
  vnl_matrix<double> Q(VDim+1, VDim+1);
  Q.set_identity();

  for(unsigned int i = 0; i < VDim; i++)
    {
    for(unsigned int j = 0; j < VDim; j++)
      {
      Q(i,j) = (double) tran->GetMatrix()(i,j);
      }
    Q(i,VDim) = (double) tran->GetOffset()[i];
    }

  WriteAffineMatrix(filename, Q);
}

template <unsigned int VDim, typename TReal>
template <class TImage>
itk::SmartPointer<TImage>
GreedyApproach<VDim, TReal>
::ReadImageViaCache(const std::string &filename,
                    itk::IOComponentEnum *comp_type)
{
  // Check the cache for the presence of the image
  typename ImageCache::const_iterator it = m_ImageCache.find(filename);
  if(it != m_ImageCache.end())
    {
    itk::Object *cached_object = it->second.target;
    TImage *image = dynamic_cast<TImage *>(cached_object);
    if(!image)
      throw GreedyException("Cached image %s cannot be cast to type %s",
                            filename.c_str(), typeid(TImage).name());
    itk::SmartPointer<TImage> pointer = image;

    // The component type is unknown here
    if(comp_type)
      *comp_type = itk::IOComponentEnum::UNKNOWNCOMPONENTTYPE;

    return pointer;
    }

  // Read the image using ITK reader
  typedef itk::ImageFileReader<TImage> ReaderType;
  typename ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(filename.c_str());
  reader->Update();

  // Store the component type if requested
  if(comp_type)
    *comp_type = reader->GetImageIO()->GetComponentType();

  itk::SmartPointer<TImage> pointer = reader->GetOutput();
  return pointer;
}

template <unsigned int VDim, typename TReal>
template <class TObject>
TObject *
GreedyApproach<VDim, TReal>
::CheckCache(const std::string &filename) const
{
  // Check the cache for the presence of the image
  typename ImageCache::const_iterator it = m_ImageCache.find(filename);
  if(it != m_ImageCache.end())
    {
    itk::Object *cached_object = it->second.target;
    return dynamic_cast<TObject *>(cached_object);
    }

  return nullptr;
}

template <unsigned int VDim, typename TReal>
typename GreedyApproach<VDim, TReal>::ImageBaseType::Pointer
GreedyApproach<VDim, TReal>
::ReadImageBaseViaCache(const std::string &filename)
{
  // Check the cache for the presence of the image
  typename ImageCache::const_iterator it = m_ImageCache.find(filename);
  if(it != m_ImageCache.end())
    {
    ImageBaseType *image_base = dynamic_cast<ImageBaseType *>(it->second.target);
    if(!image_base)
      throw GreedyException("Cached image %s cannot be cast to type %s",
                            filename.c_str(), typeid(ImageBaseType).name());
    typename ImageBaseType::Pointer pointer = image_base;
    return pointer;
    }

  // Read the image using ITK reader
  typedef itk::ImageFileReader<ImageType> ReaderType;
  typename ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(filename.c_str());
  reader->Update();

  typename ImageBaseType::Pointer pointer = reader->GetOutput();
  return pointer;
}


template <unsigned int VDim, typename TReal>
template <class TImage>
void
GreedyApproach<VDim, TReal>
::WriteImageViaCache(TImage *img, const std::string &filename, itk::IOComponentEnum comp)
{
  typename ImageCache::const_iterator it = m_ImageCache.find(filename);
  if(it != m_ImageCache.end())
    {
    // The image was found in the cache. Make sure it is an image pointer
    typename LDDMMType::ImageBaseType *cached =
        dynamic_cast<typename LDDMMType::ImageBaseType *>(it->second.target);

    if(!cached)
      throw GreedyException("Cached image %s cannot be cast to ImageBase",
                            filename.c_str(), typeid(TImage).name());

    // Try the automatic cast
    bool cast_rc = false;

    // This is a little dumb, but the vimg_write uses some special code that
    // we need to account for
    if(dynamic_cast<VectorImageType *>(img))
      cast_rc =LDDMMType::vimg_auto_cast(dynamic_cast<VectorImageType *>(img), cached);
    else if(dynamic_cast<ImageType *>(img))
      cast_rc =LDDMMType::img_auto_cast(dynamic_cast<ImageType *>(img), cached);
    else if(dynamic_cast<CompositeImageType *>(img))
      cast_rc =LDDMMType::cimg_auto_cast(dynamic_cast<CompositeImageType *>(img), cached);
    else
      {
      // Some other type (e.g., LabelImage). Instead of doing an auto_cast, we require the
      // cached object to be of the same type as the image
      TImage *cached_typed = dynamic_cast<TImage *>(cached);
      if(cached_typed)
        {
        cached_typed->CopyInformation(img);
        cached_typed->SetRegions(img->GetBufferedRegion());
        cached_typed->Allocate();
        itk::ImageAlgorithm::Copy(img, cached_typed,
                                  img->GetBufferedRegion(), cached_typed->GetBufferedRegion());
        cast_rc = true;
        }
      else throw GreedyException("Cached image %s cannot be cast to type %s",
                                 filename.c_str(), typeid(TImage).name());
      }


    // If cast failed, throw exception
    if(!cast_rc)
      throw GreedyException("Image to save %s could not cast to any known type", filename.c_str());
    }

  if(it == m_ImageCache.end() || it->second.force_write)
    {
    // This is a little dumb, but the vimg_write uses some special code that
    // we need to account for
    if(dynamic_cast<VectorImageType *>(img))
      LDDMMType::vimg_write(dynamic_cast<VectorImageType *>(img), filename.c_str(), comp);
    else if(dynamic_cast<ImageType *>(img))
      LDDMMType::img_write(dynamic_cast<ImageType *>(img), filename.c_str(), comp);
    else if(dynamic_cast<CompositeImageType *>(img))
      LDDMMType::cimg_write(dynamic_cast<CompositeImageType *>(img), filename.c_str(), comp);
    else
      {
      // Some other type (e.g., LabelImage). We use the image writer and ignore the comp
      typedef itk::ImageFileWriter<TImage> WriterType;
      typename WriterType::Pointer writer = WriterType::New();
      writer->SetFileName(filename.c_str());
      writer->SetUseCompression(true);
      writer->SetInput(img);
      writer->Update();
      }
    }
}



#include <itkBinaryErodeImageFilter.h>
#include <itkConstantPadImageFilter.h>

template <unsigned int VDim, typename TReal>
typename GreedyApproach<VDim, TReal>::CompositeImagePointer
GreedyApproach<VDim, TReal>
::ResampleImageToReferenceSpaceIfNeeded(
    CompositeImageType *img,
    ImageBaseType *ref_space,
    VectorImageType *resample_warp,
    TReal fill_value)
{
  if(LDDMMType::img_same_space(ref_space, img) && !resample_warp)
    return CompositeImagePointer(img);

  CompositeImagePointer resampled =
      LDDMMType::new_cimg(ref_space, img->GetNumberOfComponentsPerPixel());

  VectorImagePointer phi = resample_warp;
  if(!phi)
    phi = LDDMMType::new_vimg(ref_space);

  LDDMMType::interp_cimg(img, phi, resampled, false, true, fill_value);
  return resampled;
}

template<unsigned int VDim, typename TReal>
typename GreedyApproach<VDim, TReal>::ImagePointer
GreedyApproach<VDim, TReal>
::ResampleMaskToReferenceSpaceIfNeeded(
    ImageType *mask, ImageBaseType *ref_space, VectorImageType *resample_warp)
{
  if(LDDMMType::img_same_space(ref_space, mask) && !resample_warp)
    return ImagePointer(mask);

  ImagePointer resampled = LDDMMType::new_img(ref_space);

  VectorImagePointer phi = resample_warp;
  if(!phi)
    phi = LDDMMType::new_vimg(ref_space);

  LDDMMType::interp_img(mask, phi, resampled, true, true, 0);
  return resampled;
}

template <unsigned int VDim, typename TReal>
void GreedyApproach<VDim, TReal>
::ReadImages(GreedyParameters &param, OFHelperType &ofhelper,
             bool force_resample_to_fixed_space)
{
  // Read the optional reference space
  typename OFHelperType::ImageBaseType::Pointer ref_space;
  bool use_ref_space_for_moving = force_resample_to_fixed_space;
  if(param.reference_space.size())
    {
    ref_space = ReadImageViaCache<ImageType>(param.reference_space);
    use_ref_space_for_moving = true;
    }

  // Repeat for each input set
  for(const GreedyInputGroup &is : param.input_groups)
    {
    // Read the input images and stick them into an image array
    if(is.inputs.size() == 0)
      throw GreedyException("No image inputs have been specified");

    // Start an input group
    ofhelper.NewInputGroup();

    // If the parameters include a sequence of transforms, apply it first
    VectorImagePointer moving_pre_warp;

    for(unsigned int i = 0; i < is.inputs.size(); i++)
      {
      // Read fixed and moving images
      CompositeImagePointer imgFix = ReadImageViaCache<CompositeImageType>(is.inputs[i].fixed);
      CompositeImagePointer imgMov = ReadImageViaCache<CompositeImageType>(is.inputs[i].moving);

      // Check if the reference space has already been defined
      if(!ref_space)
        {
        // Reference space may involve padding the first input image
        if(param.reference_space_padding.size())
          {
          // Check the padding size
          itk::Size<VDim> pad_size;
          if(param.reference_space_padding.size() != VDim)
            throw GreedyException("Incorrect parameter to -mm-pad, should have %d elements", VDim);
          for(unsigned int d = 0; d < VDim; d++)
            pad_size[d] = param.reference_space_padding[d];

          // Apply the padding to the fixed image
          ImagePointer imgRef = LDDMMType::new_img(imgFix);
          typedef itk::ConstantPadImageFilter<ImageType, ImageType> PadFilter;
          typename PadFilter::Pointer pad = PadFilter::New();
          pad->SetInput(imgRef);
          pad->SetPadBound(pad_size);
          pad->Update();
          ref_space = pad->GetOutput();
          use_ref_space_for_moving = true;
          }
        else
          {
          ref_space = imgFix;
          }
        }

      // Once we know the reference space, we can read the moving image pre-transforms
      if(is.moving_pre_transforms.size())
        {
        ReadTransformChain(is.moving_pre_transforms, ref_space, moving_pre_warp);
        }

      // Use NaN as the outside value if we want to create a moving mask
      TReal fill_value = param.background;

      // If the reference space does not match the fixed image space, reslice the fixed image
      // to the reference space
      imgFix = ResampleImageToReferenceSpaceIfNeeded(imgFix, ref_space, nullptr, fill_value);

      // The moving image gets resampled to reference space if the reference space is specified or
      // if a moving pre-warp is specified.
      if(moving_pre_warp || use_ref_space_for_moving)
        imgMov = ResampleImageToReferenceSpaceIfNeeded(imgMov, ref_space, moving_pre_warp, fill_value);

      // Add to the helper object
      ofhelper.AddImagePair(imgFix, imgMov, is.inputs[i].weight);
      }

    if(param.fixed_mask_trim_radius.size() == VDim)
      {
      if(is.fixed_mask.size())
        throw GreedyException("Cannot specify both gradient mask and gradient mask trim radius");

      ofhelper.SetGradientMaskTrimRadius(param.fixed_mask_trim_radius);
      }

    // Read the moving-space mask
    if(is.moving_mask.size())
      {
      ImagePointer imgMovMask = ReadImageViaCache<ImageType>(is.moving_mask);
      if(moving_pre_warp || use_ref_space_for_moving)
        imgMovMask = ResampleMaskToReferenceSpaceIfNeeded(imgMovMask, ref_space, moving_pre_warp);
      ofhelper.SetMovingMask(imgMovMask);
      }

    // Set the fixed mask (distinct from gradient mask)
    if(is.fixed_mask.size())
      {
      ImagePointer imgFixMask = ReadImageViaCache<ImageType>(is.fixed_mask);
      imgFixMask = ResampleMaskToReferenceSpaceIfNeeded(imgFixMask, ref_space, nullptr);
      ofhelper.SetFixedMask(imgFixMask);
      }
    }

  // Generate the optimized composite images. For the NCC metric, we add random noise to
  // the composite images, specified in units of the interquartile intensity range.
  bool ncc_metric = param.metric == GreedyParameters::NCC || param.metric == GreedyParameters::WNCC;

  // Additive noise - needed for ncc metric over constant regions, although not sure how much
  double noise = ncc_metric ? param.ncc_noise_factor : 0.0;

  // Are we going to use masked downsampling? Currently it is only a problem for the NCC
  // metric since that metric ignores masks and treats all background as zero
  bool masked_downsampling = (param.metric != GreedyParameters::NCC);

  // Build the composite images
  ofhelper.BuildCompositeImages(noise, masked_downsampling);

  // If the metric is NCC, then also apply special processing to the gradient masks
  if(ncc_metric)
    ofhelper.DilateCompositeGradientMasksForNCC(array_caster<VDim>::to_itkSize(param.metric_radius));

  // Save the image pyramid
  if(param.flag_dump_pyramid)
    {
    for(unsigned int g = 0; g < ofhelper.GetNumberOfInputGroups(); g++)
      {
      for(unsigned int i = 0; i < ofhelper.GetNumberOfLevels(); i++)
        {
        WriteImageViaCache(ofhelper.GetFixedComposite(g, i),
                           GetDumpFile(param, "dump_pyramid_group_%02d_fixed_%02d.nii.gz", g, i));
        WriteImageViaCache(ofhelper.GetMovingComposite(g, i),
                           GetDumpFile(param, "dump_pyramid_group_%02d_moving_%02d.nii.gz", g, i));
        if(ofhelper.GetFixedMask(g, i))
          WriteImageViaCache(ofhelper.GetFixedMask(g, i),
                             GetDumpFile(param, "dump_pyramid_group_%02d_fixed_mask_%02d.nii.gz", g, i));
        if(ofhelper.GetMovingMask(g, i))
          WriteImageViaCache(ofhelper.GetMovingMask(g, i),
                             GetDumpFile(param, "dump_pyramid_group_%02d_moving_mask_%02d.nii.gz", g, i));
        }
      }
    }
}


template <unsigned int VDim, typename TReal>
vnl_matrix<double>
GreedyApproach<VDim, TReal>
::MapAffineToPhysicalRASSpace(
    OFHelperType &of_helper, unsigned int group, unsigned int level,
    LinearTransformType *tran)
{
  // Map the transform to NIFTI units
  vnl_matrix<double> T_fix, T_mov, Q, A;
  vnl_vector<double> s_fix, s_mov, p, b;

  GetVoxelSpaceToNiftiSpaceTransform(of_helper.GetReferenceSpace(level), T_fix, s_fix);
  GetVoxelSpaceToNiftiSpaceTransform(of_helper.GetMovingReferenceSpace(group, level), T_mov, s_mov);

  itk_matrix_to_vnl_matrix(tran->GetMatrix(), A);
  itk_vector_to_vnl_vector(tran->GetOffset(), b);

  Q = T_mov * A * vnl_matrix_inverse<double>(T_fix).as_matrix();
  p = T_mov * b + s_mov - Q * s_fix;

  vnl_matrix<double> Qp(VDim+1, VDim+1);
  Qp.set_identity();
  for(unsigned int i = 0; i < VDim; i++)
    {
    Qp(i, VDim) = p(i);
    for(unsigned int j = 0; j < VDim; j++)
      Qp(i,j) = Q(i,j);
    }

  return Qp;
}

template <unsigned int VDim, typename TReal>
void
GreedyApproach<VDim, TReal>
::MapPhysicalRASSpaceToAffine(
    OFHelperType &of_helper, unsigned int group, unsigned int level,
    vnl_matrix<double> &Qp,
    LinearTransformType *tran)
{
  // Map the transform to NIFTI units
  vnl_matrix<double> T_fix, T_mov, Q(VDim, VDim), A;
  vnl_vector<double> s_fix, s_mov, p(VDim), b;

  GetVoxelSpaceToNiftiSpaceTransform(of_helper.GetReferenceSpace(level), T_fix, s_fix);
  GetVoxelSpaceToNiftiSpaceTransform(of_helper.GetMovingReferenceSpace(group, level), T_mov, s_mov);

  for(unsigned int i = 0; i < VDim; i++)
    {
    p(i) = Qp(i, VDim);
    for(unsigned int j = 0; j < VDim; j++)
      Q(i,j) = Qp(i,j);
    }

  // A = vnl_matrix_inverse<double>(T_mov) * (Q * T_fix);
  // b = vnl_matrix_inverse<double>(T_mov) * (p - s_mov + Q * s_fix);
  A=vnl_svd<double>(T_mov).solve(Q * T_fix);
  b=vnl_svd<double>(T_mov).solve(p - s_mov + Q * s_fix);

  typename LinearTransformType::MatrixType tran_A;
  typename LinearTransformType::OffsetType tran_b;

  vnl_matrix_to_itk_matrix(A, tran_A);
  vnl_vector_to_itk_vector(b, tran_b);

  tran->SetMatrix(tran_A);
  tran->SetOffset(tran_b);
}

template <unsigned int VDim, typename TReal>
void
GreedyApproach<VDim, TReal>
::RecordMetricValue(const MultiComponentMetricReport &metric)
{
  if(m_MetricLog.size())
    m_MetricLog.back().push_back(metric);
}

/**
 * Find a plane of symmetry in an image
 */
/*
template <unsigned int VDim, typename TReal>
vnl_vector<double>
GreedyApproach<VDim, TReal>
::FindSymmetryPlane(ImageType *image, int N, int n_search_pts)
{
  typedef vnl_vector_fixed<double, 3> Vec3;
  typedef vnl_matrix_fixed<double, 3, 3> Mat3;

  // Loop over direction on a sphere, using the Saff & Kuijlaars algorithm
  // https://perswww.kuleuven.be/~u0017946/publications/Papers97/art97a-Saff-Kuijlaars-MI/Saff-Kuijlaars-MathIntel97.pdf
  double phi = 0.0;
  double spiral_const = 3.6 / sqrt(N);
  for(int k = 0; k < n_sphere_pts; k++)
    {
    // Height of the k-th point
    double cos_theta = -1 * (2 * k) / (N - 1);
    double sin_theta = sqrt(1 - cos_theta * cos_theta);

    // Phase of the k-th point
    if(k > 0 && k < N-1)
      phi = fmod(phi_last + spiral_const / sin_theta, vnl_math::pi * 2);
    else
      phi = 0.0;

    // We now have the polar coordinates of the points, get cartesian coordinates
    Vec3 q;
    q[0] = sin_theta * cos(phi);
    q[1] = sin_theta * sin(phi);
    q[2] = cos_theta;

    // Now q * (x,y,z) = 0 defines a plane through the origin. We will test whether the image
    // is symmetric across this plane. We first construct the reflection matrix
    Mat3 R;
    R(0,0) =  1 - q[0] * q[0]; R(0,1) = -2 * q[1] * q[0]; R(0,2) = -2 * q[2] * q[0];
    R(1,0) = -2 * q[0] * q[1]; R(1,1) =  1 - q[1] * q[1]; R(1,2) = -2 * q[2] * q[1];
    R(2,0) = -2 * q[0] * q[2]; R(2,1) = -2 * q[1] * q[2]; R(2,2) =  1 - q[2] * q[2];

    // We must find the reasonable range of intercepts to search for. An intercept is reasonable
    // if the plane cuts the image volume in at least a 80/20 ratio (let's say)


    // This is a test axis of rotation. We will now measure the symmetry of the image across this axis
    // To do so, we will construct a series of flips across this direction

    }
}
*/

/**
 * This method performs initial alignment by first searching for a plane of symmetry
 * in each image, and then finding the transform between the planes of symmetry.
 *
 * The goal is to have an almost sure-fire method for brain alignment, yet generic
 * enough to work for other data as well.
 */
/*
template <unsigned int VDim, typename TReal>
int GreedyApproach<VDim, TReal>
::SymmetrySearch(GreedyParameters &param, int level, OFHelperType *of_helper)
{

}
*/

template <unsigned int VDim, typename TReal>
AbstractAffineCostFunction<VDim, TReal> *
GreedyApproach<VDim, TReal>
::CreateAffineCostFunction(GreedyParameters &param, OFHelperType &of_helper, int level)
{
  typedef AbstractAffineCostFunction<VDim, TReal> AbstractAffineCostFunction;
  typedef RigidCostFunction<VDim, TReal> RigidCostFunction;
  typedef ScalingCostFunction<VDim, TReal> ScalingCostFunction;
  typedef PhysicalSpaceAffineCostFunction<VDim, TReal> PhysicalSpaceAffineCostFunction;
  typedef MaskWeightedSumAffineConstFunction<VDim, TReal> CompositeCostFunction;

  // Create a list of individual cost functions, on for each input group
  std::vector<AbstractAffineCostFunction *> components;
  for(unsigned int g = 0; g < of_helper.GetNumberOfInputGroups(); g++)
    {
    // Define the affine cost function
    if(param.affine_dof == GreedyParameters::DOF_RIGID)
      {
      RigidCostFunction *rigid_acf =
          new RigidCostFunction(&param, this, g, level, &of_helper);
      components.push_back(
            new ScalingCostFunction(
              rigid_acf,
              rigid_acf->GetOptimalParameterScaling(
                of_helper.GetReferenceSpace(level)->GetBufferedRegion().GetSize())));
      }
    else
      {
      //  PureAffineCostFunction *affine_acf = new PureAffineCostFunction(&param, level, &of_helper);
      PhysicalSpaceAffineCostFunction *affine_acf =
          new PhysicalSpaceAffineCostFunction(&param, this, g, level, &of_helper);
      components.push_back(
            new ScalingCostFunction(
              affine_acf,
              affine_acf->GetOptimalParameterScaling(
                of_helper.GetReferenceSpace(level)->GetBufferedRegion().GetSize())));
      }
    }

  // Create a function that incorporates them all
  return new CompositeCostFunction(components);
}

template <unsigned int VDim, typename TReal>
void
GreedyApproach<VDim, TReal>
::InitializeAffineTransform(GreedyParameters &param, OFHelperType &of_helper,
                            AbstractAffineCostFunction<VDim, TReal> *acf,
                            LinearTransformType *tLevel)
{
  typedef AbstractAffineCostFunction<VDim, TReal> AbstractAffineCostFunction;
  typedef RigidCostFunction<VDim, TReal> RigidCostFunction;
  typedef ScalingCostFunction<VDim, TReal> ScalingCostFunction;
  typedef PhysicalSpaceAffineCostFunction<VDim, TReal> PhysicalSpaceAffineCostFunction;
  typedef MaskWeightedSumAffineConstFunction<VDim, TReal> GroupAffineCostFunction;

  // Get the coefficients corresponding to the identity transform in voxel space
  tLevel->SetIdentity();
  vnl_vector<double> xIdent = acf->GetCoefficients(tLevel);

  // Use the provided initial affine as the starting point
  if(param.affine_init_mode == RAS_FILENAME)
    {
    // Read the initial affine transform from a file
    vnl_matrix<double> Qp = this->ReadAffineMatrixViaCache(param.affine_init_transform);

    // Map this to voxel space
    // TODO: this does not make any sense, really... Should change all affine ops to work in physical space
    MapPhysicalRASSpaceToAffine(of_helper, 0, 0, Qp, tLevel);
    }
  else if(param.affine_init_mode == RAS_IDENTITY)
    {
    // Physical space transform
    vnl_matrix<double> Qp(VDim+1, VDim+1); Qp.set_identity();

    // Map this to voxel space
    // TODO: this does not make any sense, really... Should change all affine ops to work in physical space
    MapPhysicalRASSpaceToAffine(of_helper, 0, 0, Qp, tLevel);
    }
  else if(param.affine_init_mode == IMG_CENTERS)
    {
    // Find a translation that maps center voxel of fixed image to the center
    // voxel of the moving image
    vnl_matrix<double> Qp(VDim+1, VDim+1); Qp.set_identity();
    vnl_vector<double> cfix = GetImageCenterinNiftiSpace(of_helper.GetReferenceSpace(0));
    vnl_vector<double> cmov = GetImageCenterinNiftiSpace(of_helper.GetMovingReferenceSpace(0, 0));

    // TODO: I think that setting the matrix above to affine will break the registration
    // if fixed and moving are in different orientations? Or am I crazy?

    // Compute the transform that takes fixed into moving
    for(unsigned int d = 0; d < VDim; d++)
      Qp(d, VDim) = cmov[d] - cfix[d];

    // Map this to voxel space
    // TODO: this does not make any sense, really... Should change all affine ops to work in physical space
    MapPhysicalRASSpaceToAffine(of_helper, 0, 0, Qp, tLevel);
    }

  // Get the new coefficients
  vnl_vector<double> xInit = acf->GetCoefficients(tLevel);

  // If the voxel-space transform is identity, apply a little bit of jitter
  if((xIdent - xInit).inf_norm() < 1e-4)
    {
    // Apply jitter
    vnl_random rndy(12345);
    for (unsigned i = 0; i < xInit.size(); i++)
      xInit[i] += rndy.drand32(-0.4, 0.4);

    // Map back into transform format
    acf->GetTransform(xInit, tLevel);
    }

  // If the uses asks for rigid search, do it!
  if(param.rigid_search.iterations > 0)
    {
    // Random seed. TODO: let user supply seed
    vnl_random randy(12345);

    // For rigid search, we must search in physical space, rather than in voxel space.
    // This is the affine transformation in physical space that corresponds to whatever
    // the current initialization is.
    // TODO: this does not make any sense, really... Should change all affine ops to work in physical space
    vnl_matrix<double> Qp = MapAffineToPhysicalRASSpace(of_helper, 0, 0, tLevel);

    // Get the center of the fixed image in physical coordinates
    vnl_vector<double> cfix = GetImageCenterinNiftiSpace(of_helper.GetReferenceSpace(0));

    // Create a pure rigid acf for each group
    std::vector<AbstractAffineCostFunction *> group_search_fn_list;
    for(unsigned int g = 0; g < of_helper.GetNumberOfInputGroups(); g++)
      group_search_fn_list.push_back(new RigidCostFunction(&param, this, g, 0, &of_helper));

    // Create a group acf that integrates over all groups
    GroupAffineCostFunction search_fun(group_search_fn_list);

    // Report the initial best
    double fBest = 0.0;
    vnl_vector<double> xBest = search_fun.GetCoefficients(tLevel);
    search_fun.compute(xBest, &fBest, NULL);
    std::cout << "Rigid search -> Initial best: " << fBest << " " << xBest << std::endl;

    // Loop over random iterations
    for(int i = 0; i < param.rigid_search.iterations; i++)
      {
      // Depending on the search mode, we either apply a small rotation, or any random rotation,
      // or a random rotation and a flip to the input. Whatever rotation we apply, it must
      // be around the center of the fixed coordinate system.
      typename RigidCostFunction::Mat RF;
      if(param.rigid_search.mode == RANDOM_NORMAL_ROTATION)
        {
        // Random angle in radians
        double alpha = randy.normal() * param.rigid_search.sigma_angle * 0.01745329252;
        RF = RigidCostFunction::GetRandomRotation(randy, alpha);
        }
      else if(param.rigid_search.mode == ANY_ROTATION)
        {
        double alpha = randy.drand32(-vnl_math::pi, vnl_math::pi);
        RF = RigidCostFunction::GetRandomRotation(randy, alpha);
        }
      else if(param.rigid_search.mode == ANY_ROTATION_AND_FLIP)
        {
        typename RigidCostFunction::Mat R, F;
        F.set_identity();
        for(unsigned int a = 0; a < VDim; a++)
          F(a,a) = (randy.normal() > 0.0) ? 1.0 : -1.0;
        double alpha = randy.drand32(-vnl_math::pi, vnl_math::pi);
        R = RigidCostFunction::GetRandomRotation(randy, alpha);
        RF = R * F;
        }
      else throw GreedyException("Unknown rotation search mode encountered");

      // Find the offset so that the rotation/flip preserve fixed image center
      typename RigidCostFunction::Vec b_RF = cfix - RF * cfix;

      // Create the physical space matrix corresponding to random search point
      vnl_matrix<double> Qp_rand(VDim+1, VDim+1); Qp_rand.set_identity();
      Qp_rand.update(RF.as_matrix());
      for(unsigned int a = 0; a < VDim; a++)
        Qp_rand(a,VDim) = b_RF[a];

      // Combine the two matrices. The matrix Qp_rand operates in fixed image space so
      // it should be applied first, followed by Qp
      vnl_matrix<double> Qp_search = Qp * Qp_rand;

      // Add the random translation
      for(unsigned int a = 0; a < VDim; a++)
        Qp_search(a,VDim) += randy.normal() * param.rigid_search.sigma_xyz;

      // Convert this physical space transformation into a voxel-space transform
      typename LinearTransformType::Pointer tSearchTry = LinearTransformType::New();

      // TODO: this conversion to voxel space is all wrong
      MapPhysicalRASSpaceToAffine(of_helper, 0, 0, Qp_search, tSearchTry);

      // Evaluate the metric for this point
      vnl_vector<double> xTry = search_fun.GetCoefficients(tSearchTry);
      double f = 0.0;
      search_fun.compute(xTry, &f, NULL);

      // Is this an improvement?
      if(f < fBest)
        {
        fBest = f;
        tLevel->SetMatrix(tSearchTry->GetMatrix());
        tLevel->SetOffset(tSearchTry->GetOffset());
        std::cout << "Rigid search -> Iter " << i << ": " << fBest << " "
                  << xTry << " det = " << vnl_determinant(Qp_search)
                  <<  std::endl;
        }
      }
    }
}

template <unsigned int VDim, typename TReal>
int GreedyApproach<VDim, TReal>
::CheckAffineDerivatives(GreedyParameters &param, OFHelperType &of_helper,
                         AbstractAffineCostFunction<VDim, TReal> *acf,
                         LinearTransformType *tLevel, int level, double tol)
{
  // Test the gradient computation
  vnl_vector<double> xLevel = acf->GetCoefficients(tLevel);
  vnl_vector<double> xGrad(acf->get_number_of_unknowns(), 0.0);
  double f0;
  acf->compute(xLevel, &f0, &xGrad);

  // Propagate the jitter to the transform
  // TODO: this conversion to voxel space is all wrong
  vnl_matrix<double> Q_physical = MapAffineToPhysicalRASSpace(of_helper, 0, level, tLevel);
  std::cout << "Initial RAS Transform: " << std::endl << Q_physical  << std::endl;

  printf("*** Affine Derivative Check ***\n");
  printf("ANL gradient: ");
  for (unsigned i = 0; i < xGrad.size(); i++)
    printf("%11.4f ", xGrad[i]);
  printf("\n");

  vnl_vector<double> xGradN(acf->get_number_of_unknowns(), 0.0);
  int status = 0;
  for(int i = 0; i < acf->get_number_of_unknowns(); i++)
    {
    // double eps = (i % VDim == 0) ? 1.0e-2 : 1.0e-5;
    double eps = param.deriv_epsilon;
    double f[] = {0., 0., 0., 0.};
    vnl_vector<double> x[] = {xLevel, xLevel, xLevel, xLevel};
    x[0][i] -= 2 * eps; x[1][i] -= eps; x[2][i] += eps; x[3][i] += 2 * eps;

    // Keep track of gradient even though we do not need it. There is an apparent bug
    // at least with the NCC metric, where the reuse of the working image in a scenario
    // where you first compute the gradient and then do not, causes the iteration through
    // the working image to incorrectly align the per-pixel arrays. Asking for gradient
    // every time is a little more costly, but it avoids this issue
    vnl_vector<double> xGradDummy(acf->get_number_of_unknowns(), 0.0);

    // Four-point derivative computation
    for(unsigned int j = 0; j < 4; j++)
      {
      acf->compute(x[j], &f[j], &xGradDummy);

      // Uncomment this if you really want to debug these derivatives!
      /*
      ImageType *metric = acf->GetMetricImage();
      char buffer[256];
      sprintf(buffer, "/tmp/grad_metric_param_%02d_off_%d.nii.gz", i, j);
      LDDMMType::img_write(metric, buffer);
      */
      }

    xGradN[i] = (f[0] - 8 * f[1] + 8 * f[2] - f[3]) / (12 * eps);

    if(fabs(xGrad[i] - xGradN[i]) > tol)
      status = -1;
    }

  printf("NUM gradient: ");
  for (unsigned i = 0; i < xGradN.size(); i++)
    printf("%11.4f ", xGradN[i]);
  printf("\n");

  // Print the matrix components and b components
  printf("\n     ");
  for(unsigned int a = 0; a < VDim; a++)
    for(unsigned int b = 0; b < VDim; b++)
      printf("      A_%d%d", a, b);
  for(unsigned int a = 0; a < VDim; a++)
    printf("       b_%d", a);

  // Print the two matrices
  printf("\nANL:  ");
  acf->GetTransform(xGrad, tLevel);
  for(unsigned int a = 0; a < VDim; a++)
    for(unsigned int b = 0; b < VDim; b++)
      printf("%9.4f ", tLevel->GetMatrix()(a,b));
  for(unsigned int a = 0; a < VDim; a++)
    printf("%9.4f ", tLevel->GetOffset()[a]);

  printf("\nNUM:  ");
  acf->GetTransform(xGradN, tLevel);
  for(unsigned int a = 0; a < VDim; a++)
    for(unsigned int b = 0; b < VDim; b++)
      printf("%9.4f ", tLevel->GetMatrix()(a,b));
  for(unsigned int a = 0; a < VDim; a++)
    printf("%9.4f ", tLevel->GetOffset()[a]);
  printf("\n\n");

  return status;
}

template<unsigned int VDim, typename TReal>
std::string
GreedyApproach<VDim, TReal>
::GetDumpFile(const GreedyParameters &param, const char *pattern, ...)
{
  // Fill out the pattern with sprintf-like parameters
  char buffer[4096];
  va_list args;
  va_start (args, pattern);
  vsprintf (buffer,pattern, args);
  va_end (args);

  // Prepend dump path
  std::string full_path = param.dump_prefix + buffer;
  std::string dump_dir = itksys::SystemTools::GetFilenamePath(full_path);
  if(dump_dir.size())
    itksys::SystemTools::MakeDirectory(dump_dir);

  // Return the filename
  return full_path;
}


template <unsigned int VDim, typename TReal>
int GreedyApproach<VDim, TReal>
::RunAffine(GreedyParameters &param)
{
  typedef AbstractAffineCostFunction<VDim, TReal> AbstractAffineCostFunction;
  typedef RigidCostFunction<VDim, TReal> RigidCostFunction;
  typedef ScalingCostFunction<VDim, TReal> ScalingCostFunction;
  typedef PhysicalSpaceAffineCostFunction<VDim, TReal> PhysicalSpaceAffineCostFunction;

  // Create an optical flow helper object
  OFHelperType of_helper;

  // Object for text output
  GreedyStdOut gout(param.verbosity);

  // Set the scaling factors for multi-resolution
  of_helper.SetDefaultPyramidFactors(param.iter_per_level.size());

  // Add random sampling jitter for affine stability at voxel edges
  of_helper.SetJitterSigma(param.affine_jitter);

  // Read the image pairs to register - this will also build the composite pyramids
  // In affine mode, we do not force resampling of moving image to fixed image space
  ReadImages(param, of_helper, false);

  // Matrix describing current transform in physical space
  vnl_matrix<double> Q_physical;

  // The number of resolution levels
  unsigned nlevels = param.iter_per_level.size();

  // Clear the metric log
  m_MetricLog.clear();

  // Iterate over the resolution levels
  for(unsigned int level = 0; level < nlevels; ++level)
    {
    // Add stage to metric log
    m_MetricLog.push_back(std::vector<MultiComponentMetricReport>());

    // Define the affine cost function
    AbstractAffineCostFunction *acf = CreateAffineCostFunction(param, of_helper, level);

    // Current transform
    typename LinearTransformType::Pointer tLevel = LinearTransformType::New();

    // Set up the initial transform
    if(level == 0)
      {
      // Use parameters to load initial transform
      InitializeAffineTransform(param, of_helper, acf, tLevel);
      }
    else
      {
      // Update the transform from the last level
      // TODO: this does not make any sense, really... Should change all affine ops to work in physical space
      MapPhysicalRASSpaceToAffine(of_helper, 0, level, Q_physical, tLevel);
      }

    // Test derivatives
    // Convert to a parameter vector
    vnl_vector<double> xLevel = acf->GetCoefficients(tLevel.GetPointer());

    if(param.flag_debug_deriv)
      {
      CheckAffineDerivatives(param, of_helper, acf, tLevel, level, 1e-6);
      }

    // Run the minimization
    if(param.iter_per_level[level] > 0)
      {
      if(param.flag_powell)
        {
        // Set up the optimizer
        vnl_powell *optimizer = new vnl_powell(acf);
        optimizer->set_f_tolerance(1e-9);
        optimizer->set_x_tolerance(1e-4);
        optimizer->set_trace(param.verbosity > GreedyParameters::VERB_NONE);
        optimizer->set_verbose(param.verbosity > GreedyParameters::VERB_DEFAULT);
        optimizer->set_max_function_evals(param.iter_per_level[level]);

        optimizer->minimize(xLevel);
        delete optimizer;

        }
      else
        {
        // Set up the optimizer
        vnl_lbfgs *optimizer = new vnl_lbfgs(*acf);

        // Using defaults from scipy
        double ftol = (param.lbfgs_param.ftol == 0.0) ? 2.220446049250313e-9 : param.lbfgs_param.ftol;
        double gtol = (param.lbfgs_param.gtol == 0.0) ? 1e-05 : param.lbfgs_param.gtol;

        optimizer->set_f_tolerance(ftol);
        optimizer->set_g_tolerance(gtol);
        if(param.lbfgs_param.memory > 0)
          optimizer->memory = param.lbfgs_param.memory;

        optimizer->set_trace(param.verbosity > GreedyParameters::VERB_NONE);
        optimizer->set_verbose(param.verbosity > GreedyParameters::VERB_DEFAULT);
        optimizer->set_max_function_evals(param.iter_per_level[level]);

        /*
        vnl_conjugate_gradient *optimizer = new vnl_conjugate_gradient(*acf);
        optimizer->set_trace(param.verbosity > GreedyParameters::VERB_NONE);
        optimizer->set_verbose(param.verbosity > GreedyParameters::VERB_DEFAULT);
        optimizer->set_max_function_evals(param.iter_per_level[level]);
        */

        optimizer->minimize(xLevel);
        delete optimizer;
        }

      if(param.flag_debug_aff_obj && param.iter_per_level[level] > 0)
        {
        for(int k = -10; k < 10; k++)
          {
          printf("Obj\t%d\t", k);
          for(int i = 0; i < acf->get_number_of_unknowns(); i++)
            {
            vnl_vector<double> xTest = xLevel;
            xTest[i] = xLevel[i] + k * param.deriv_epsilon;
            double f; acf->compute(xTest, &f, NULL);
            printf("%12.8f\t", f);

            ImageType *metric = acf->GetMetricImage();
            char buffer[256];
            sprintf(buffer, "/tmp/debug_aff_obj_%03d_par_%02d.nii.gz", k, i);
            LDDMMType::img_write(metric, buffer);
            }
          printf("\n");
          }
          {
          vnl_vector<double> xTest = xLevel;
            {
            }
          printf("\n");
          }
        }

      // Did the registration succeed?
      if(xLevel.size() > 0)
        {
        // Get the final transform
        typename LinearTransformType::Pointer tFinal = LinearTransformType::New();
        acf->GetTransform(xLevel, tFinal.GetPointer());

        // TODO: this does not make any sense, really... Should change all affine ops to work in physical space
        Q_physical = MapAffineToPhysicalRASSpace(of_helper, 0, level, tFinal);
        }
      else
        {
        // Use the pre-initialization transform parameters
        // TODO: this does not make any sense, really... Should change all affine ops to work in physical space
        Q_physical = MapAffineToPhysicalRASSpace(of_helper, 0, level, tLevel);
        }

      // End of level report
      gout.printf("END OF LEVEL %3d\n", level);

      // Print final metric report
      MultiComponentMetricReport metric_report = this->GetMetricLog()[level].back();
      gout.printf("Level %3d  LastIter   Metrics", level);
      for (unsigned i = 0; i < metric_report.ComponentPerPixelMetrics.size(); i++)
        gout.printf("  %8.6f", metric_report.ComponentPerPixelMetrics[i]);
      gout.printf("  Energy = %8.6f\n", metric_report.TotalPerPixelMetric);
      gout.flush();
      }

    // Print the final RAS transform for this level (even if no iter)
    gout.printf("Level %3d  Final RAS Transform:\n", level);
    for(unsigned int a = 0; a < VDim+1; a++)
      {
      for(unsigned int b = 0; b < VDim+1; b++)
        gout.printf("%8.4f%c", Q_physical(a,b), b < VDim ? ' ' : '\n');
      }

    delete acf;
    }

  // Write the final affine transform
  this->WriteAffineMatrixViaCache(param.output, Q_physical);
  return 0;
}





#include "itkStatisticsImageFilter.h"

/** My own time probe because itk's use of fork is messing up my debugging */
class GreedyTimeProbe
{
public:
  GreedyTimeProbe();
  void Start();
  void Stop();
  double GetMean() const;
  double GetTotal() const;
protected:
  double m_TotalTime;
  double m_StartTime;
  unsigned long m_Runs;
};

GreedyTimeProbe::GreedyTimeProbe()
{
  m_TotalTime = 0.0;
  m_StartTime = 0.0;
  m_Runs = 0.0;
}

void GreedyTimeProbe::Start()
{
  m_StartTime = clock();
}

void GreedyTimeProbe::Stop()
{
  if(m_StartTime == 0.0)
    throw GreedyException("Timer stop without start");
  m_TotalTime += clock() - m_StartTime;
  m_StartTime = 0.0;
  m_Runs++;
}

double GreedyTimeProbe::GetMean() const
{
  if(m_Runs == 0)
    return 0.0;
  else
    return m_TotalTime / (CLOCKS_PER_SEC * m_Runs);
}

double GreedyTimeProbe::GetTotal() const
{
  return m_TotalTime / CLOCKS_PER_SEC;
}

template <unsigned int VDim, typename TReal>
std::string
GreedyApproach<VDim, TReal>
::PrintIter(int level, int iter, const MultiComponentMetricReport &metric) const
{
  // Start with a buffer
  char b_level[64], b_iter[64], b_metrics[512], b_line[1024];

  if(level < 0)
    sprintf(b_level, "LastLevel");
  else
    sprintf(b_level, "Level %03d", level);

  if(iter < 0)
    sprintf(b_iter, "LastIter");
  else
    sprintf(b_iter, "Iter %05d", iter);

  if(metric.ComponentPerPixelMetrics.size() > 1)
    {
    int pos = sprintf(b_metrics, "Metrics");
    for (unsigned i = 0; i < metric.ComponentPerPixelMetrics.size(); i++)
      pos += sprintf(b_metrics + pos, "  %8.6f", metric.ComponentPerPixelMetrics[i]);
    }
  else
    sprintf(b_metrics, "");

  sprintf(b_line, "%s  %s  %s  Energy = %8.6f", b_level, b_iter, b_metrics, metric.TotalPerPixelMetric);
  std::string result = b_line;

  return b_line;
}

template <unsigned int VDim, typename TReal>
void GreedyApproach<VDim, TReal>
::EvaluateMetricForDeformableRegistration(
    GreedyParameters &param, OFHelperType &of_helper,
    unsigned int level,
    VectorImageType *phi,
    MultiComponentMetricReport &metric_report,
    ImageType *out_metric_image,
    VectorImageType *out_metric_gradient,
    double eps)
{
  // Initialize the metric and gradient to zeros
  out_metric_image->FillBuffer(0.0);
  out_metric_gradient->FillBuffer(typename VectorImageType::PixelType(0.0));

  // Reset the metric report
  metric_report = MultiComponentMetricReport();

  // Compute the individual metrics, adding to the metric image
  for(unsigned int g = 0; g < of_helper.GetNumberOfInputGroups(); g++)
    {
    // Keep track of the metric report for this group
    MultiComponentMetricReport group_report;

    // Switch based on the metric
    if(param.metric == GreedyParameters::SSD)
      {
      of_helper.ComputeSSDMetricAndGradient(g, level, phi,
                                            std::isnan(param.background),
                                            param.background,
                                            out_metric_image,
                                            group_report, out_metric_gradient, eps);
      group_report.Scale(1.0 / eps);
      }

    else if(param.metric == GreedyParameters::MI || param.metric == GreedyParameters::NMI)
      {
      of_helper.ComputeNMIMetricAndGradient(g, level, param.metric == GreedyParameters::NMI,
                                            phi, out_metric_image, group_report,
                                            out_metric_gradient, eps);

      // If there is a mask, multiply the gradient by the mask
      if(of_helper.GetFixedMask(g, level))
        LDDMMType::vimg_multiply_in_place(out_metric_gradient, of_helper.GetFixedMask(g, level));
      }

    else if(param.metric == GreedyParameters::NCC)
      {
      itk::Size<VDim> radius = array_caster<VDim>::to_itkSize(param.metric_radius);

      // Compute the metric - no need to multiply by the mask, this happens already in the NCC metric code
      of_helper.ComputeNCCMetricAndGradient(g, level, phi, radius, false,
                                            out_metric_image, group_report, out_metric_gradient, eps);
      group_report.Scale(1.0 / eps);
      }

    else if(param.metric == GreedyParameters::WNCC)
      {
      itk::Size<VDim> radius = array_caster<VDim>::to_itkSize(param.metric_radius);

      // Compute the metric - no need to multiply by the mask, this happens already in the NCC metric code
      // TODO: configure weighting
      of_helper.ComputeNCCMetricAndGradient(g, level, phi, radius, true,
                                            out_metric_image, group_report, out_metric_gradient, eps);
      group_report.Scale(1.0 / eps);
      }

    else if(param.metric == GreedyParameters::MAHALANOBIS)
      {
      of_helper.ComputeMahalanobisMetricImage(g, level, phi, out_metric_image,
                                              metric_report, out_metric_gradient);
      }

    // Append the metric report
    metric_report.Append(group_report);
    }
}

template <unsigned int VDim, typename TReal>
void GreedyApproach<VDim, TReal>
::LoadInitialTransform(
    GreedyParameters &param, OFHelperType &of_helper,
    unsigned int level, VectorImageType *phi)
{
  if(param.initial_warp.size())
    {
    // The user supplied an initial warp or initial root warp. In this case, we
    // do not start iteration from zero, but use the initial warp to start from
    VectorImagePointer uInit = VectorImageType::New();

    // Read the warp file
    LDDMMType::vimg_read(param.initial_warp.c_str(), uInit );

    // Convert the warp file into voxel units from physical units
    OFHelperType::PhysicalWarpToVoxelWarp(uInit, uInit, uInit);

    // Scale the initial warp by the pyramid level
    LDDMMType::vimg_resample_identity(uInit, of_helper.GetReferenceSpace(level), phi);
    LDDMMType::vimg_scale_in_place(phi, 1.0 / (1 << level));
    }
  else if(param.affine_init_mode != VOX_IDENTITY)
    {
    typename LinearTransformType::Pointer tran = LinearTransformType::New();

    if(param.affine_init_mode == RAS_FILENAME)
      {
      // Read the initial affine transform from a file
      vnl_matrix<double> Qp = ReadAffineMatrixViaCache(param.affine_init_transform);

      // Map this to voxel space
      // TODO: this does not make any sense, really... Should change all affine ops to work in physical space
      MapPhysicalRASSpaceToAffine(of_helper, 0, level, Qp, tran);
      }
    else if(param.affine_init_mode == RAS_IDENTITY)
      {
      // Physical space transform
      vnl_matrix<double> Qp(VDim+1, VDim+1); Qp.set_identity();

      // Map this to voxel space
      // TODO: this does not make any sense, really... Should change all affine ops to work in physical space
      MapPhysicalRASSpaceToAffine(of_helper, 0, level, Qp, tran);
      }

    // Create an initial warp
    OFHelperType::AffineToField(tran, phi);
    }
}

/**
 * This is the main function of the GreedyApproach algorithm
 */
template <unsigned int VDim, typename TReal>
int GreedyApproach<VDim, TReal>
::RunDeformable(GreedyParameters &param)
{
  // Create an optical flow helper object
  OFHelperType of_helper;

  // Object for text output
  GreedyStdOut gout(param.verbosity);

  // Set the scaling factors for multi-resolution
  of_helper.SetDefaultPyramidFactors(param.iter_per_level.size());

  // Set the scaling mode depending on the metric
  if(param.metric == GreedyParameters::MAHALANOBIS)
    of_helper.SetScaleFixedImageWithVoxelSize(true);

  // Read the image pairs to register
  // In deformable mode, we force resampling of moving image to fixed image space
  ReadImages(param, of_helper, true);

  // An image pointer desribing the current estimate of the deformation
  VectorImagePointer uLevel = nullptr;

  // The number of resolution levels
  unsigned nlevels = param.iter_per_level.size();

  // Clear the metric log
  m_MetricLog.clear();

  // Iterate over the resolution levels
  for(unsigned int level = 0; level < nlevels; ++level)
    {
    // Add stage to metric log
    m_MetricLog.push_back(std::vector<MultiComponentMetricReport>());

    // Reference space
    ImageBaseType *refspace = of_helper.GetReferenceSpace(level);

    // Smoothing factors for this level, in physical units
    typename LDDMMType::Vec sigma_pre_phys =
        of_helper.GetSmoothingSigmasInPhysicalUnits(level, param.sigma_pre.sigma,
                                                    param.sigma_pre.physical_units);

    typename LDDMMType::Vec sigma_post_phys =
        of_helper.GetSmoothingSigmasInPhysicalUnits(level, param.sigma_post.sigma,
                                                    param.sigma_post.physical_units);

    // Report the smoothing factors used
    gout.printf("LEVEL %d of %d\n", level+1, nlevels);
    gout.printf("  Smoothing sigmas (mm):");
    for(unsigned int d = 0; d < VDim; d++)
      gout.printf("%s%f", d==0 ? " " : "x", sigma_pre_phys[d]);
    for(unsigned int d = 0; d < VDim; d++)
      gout.printf("%s%f", d==0 ? " " : "x", sigma_post_phys[d]);
    gout.printf("\n");

    // Set up timers for different critical components of the optimization
    GreedyTimeProbe tm_Gradient, tm_Gaussian1, tm_Gaussian2, tm_Iteration,
      tm_Integration, tm_Update, tm_UpdatePDE, tm_PDE;

    // Intermediate images
    ImagePointer iTemp = ImageType::New();
    VectorImagePointer viTemp = VectorImageType::New();
    VectorImagePointer uk = VectorImageType::New();
    VectorImagePointer uk1 = VectorImageType::New();

    // This is the exponentiated uk, in stationary velocity mode it is uk^(2^N)
    VectorImagePointer uk_exp = VectorImageType::New();

    // A pointer to the full warp image - either uk in greedy mode, or uk_exp in diff demons mdoe
    VectorImageType *uFull;

    // Matrix work image (for Lie Bracket)
    typedef typename LDDMMType::MatrixImageType MatrixImageType;
    typename MatrixImageType::Pointer work_mat = MatrixImageType::New();

    // Sparse solver for incompressibility mode
    void *incompressibility_solver = nullptr;

    // Mask used for incompressibility purposes
    ImagePointer incompressibility_mask = nullptr;

    // Allocate the intermediate data
    LDDMMType::alloc_vimg(uk, refspace);
    if(param.iter_per_level[level] > 0)
      {
      LDDMMType::alloc_img(iTemp, refspace);
      LDDMMType::alloc_vimg(viTemp, refspace);
      LDDMMType::alloc_vimg(uk1, refspace);

      // These are only allocated in diffeomorphic demons mode
      if(param.flag_stationary_velocity_mode)
        {
        LDDMMType::alloc_vimg(uk_exp, refspace);
        LDDMMType::alloc_mimg(work_mat, refspace);
        }

      if(param.flag_stationary_velocity_mode && param.flag_incompressibility_mode)
        {
        if(of_helper.GetFixedMask(0, level))
          {
          // TODO: not sure that just using the fixed mask is right for this
          std::cout << "Setting up incompressibility mask" << std::endl;
          incompressibility_mask = LDDMMType::new_img(of_helper.GetFixedMask(0, level));
          LDDMMType::img_copy(of_helper.GetFixedMask(0, level), incompressibility_mask);
          LDDMMType::img_threshold_in_place(incompressibility_mask, 0.9, 1.0, 1.0, 0.0);
          }

        std::cout << "Setting up incompressibility solver" << std::endl;
        incompressibility_solver = LDDMMType::poisson_pde_zero_boundary_initialize(uk, incompressibility_mask);
        }
      }

    // Initialize the deformation field from initial transform or from last iteration
    if(uLevel.IsNotNull())
      {
      LDDMMType::vimg_resample_identity(uLevel, refspace, uk);
      LDDMMType::vimg_scale_in_place(uk, 2.0);
      uLevel = uk;
      }
    else
      {
      this->LoadInitialTransform(param, of_helper, level, uk);
      uLevel = uk;
      }

    // Iterate for this level
    for(int iter = 0; iter < param.iter_per_level[level]; iter++)
      {
      // Does a debug dump get generated on this iteration?
      bool flag_dump = param.flag_dump_moving && 0 == iter % param.dump_frequency;

      // Start the iteration timer
      tm_Iteration.Start();

      // The epsilon for this level
      double eps = param.epsilon_per_level[level];

      // Integrate the total deformation field for this iteration
      if(param.flag_stationary_velocity_mode)
        {
        tm_Integration.Start();

        // This is the exponentiation of the stationary velocity field
        // Take current warp to 'exponent' power - this is the actual warp
        LDDMMType::vimg_exp(uk, uk_exp, viTemp, param.warp_exponent, 1.0);
        uFull = uk_exp;

        tm_Integration.Stop();
        }
      else
        {
        uFull = uk;
        }

      // Create a metric report that will be returned by all metrics
      MultiComponentMetricReport metric_report;

      // Begin gradient computation
      tm_Gradient.Start();

      // Evaluate the correct metric
      this->EvaluateMetricForDeformableRegistration(param, of_helper, level, uFull, metric_report, iTemp, uk1, eps);

      // End gradient computation
      tm_Gradient.Stop();

      // Print a report for this iteration
      std::cout << this->PrintIter(level, iter, metric_report) << std::endl;
      fflush(stdout);

      // Record the metric value in the log
      this->RecordMetricValue(metric_report);

      // Dump the gradient image if requested
      if(flag_dump)
        {
        WriteImageViaCache(iTemp.GetPointer(), GetDumpFile(param, "dump_metric_lev%02d_iter%04d.nii.gz", level, iter));
        WriteImageViaCache(uk1.GetPointer(), GetDumpFile(param, "dump_gradient_lev%02d_iter%04d.nii.gz", level, iter));
        }

      // We have now computed the gradient vector field. Next, we smooth it
      tm_Gaussian1.Start();

      // Why do we smooth with a border? What if there is data at the border?
      // TODO: revisit smoothing around mask, think it through!
      // --- LDDMMType::vimg_smooth_withborder(uk1, viTemp, sigma_pre_phys, 1);
      LDDMMType::vimg_smooth(uk1, viTemp, sigma_pre_phys);
      tm_Gaussian1.Stop();

      // After smoothing, compute the maximum vector norm and use it as a normalizing
      // factor for the displacement field
      if(param.time_step_mode == GreedyParameters::SCALE)
        LDDMMType::vimg_normalize_to_fixed_max_length(viTemp, iTemp, eps, false);
      else if (param.time_step_mode == GreedyParameters::SCALEDOWN)
        LDDMMType::vimg_normalize_to_fixed_max_length(viTemp, iTemp, eps, true);

      // Dump the smoothed gradient image if requested
      if(flag_dump)
        WriteImageViaCache(viTemp.GetPointer(), GetDumpFile(param, "dump_optflow_lev%02d_iter%04d.nii.gz", level, iter));

      // Compute the updated deformation field - in uk1
      tm_Update.Start();
      if(param.flag_stationary_velocity_mode)
        {
        // this is diffeomorphic demons - Vercauteren 2008
        // We now hold the update field in viTemp. This update u should be integrated
        // with the current stationary velocity field such that exp[v'] = exp[v] o exp[u]
        // Vercauteren (2008) suggests using the following expressions
        // v' = v + u (so-so)
        // v' = v + u + [v, u]/2 (this is the Lie bracket)

        // Scale the update by 1 / 2^exponent (tiny update, first order approximation)
        LDDMMType::vimg_scale_in_place(viTemp, 1.0 / (2 << param.warp_exponent));

        // Use appropriate update
        if(param.flag_stationary_velocity_mode_use_lie_bracket)
          {
          // Use the Lie Bracket approximation (v + u + [v,u])
          LDDMMType::lie_bracket(uk, viTemp, work_mat, uk1);
          LDDMMType::vimg_scale_in_place(uk1, 0.5);
          LDDMMType::vimg_add_in_place(uk1, uk);
          LDDMMType::vimg_add_in_place(uk1, viTemp);
          }
        else
          {
          LDDMMType::vimg_copy(uk, uk1);
          LDDMMType::vimg_add_in_place(uk1, viTemp);
          }
        }
      else
        {
        // This is compositive (uk1 = viTemp + uk o viTemp), which is what is done with
        // compositive demons and ANTS
        LDDMMType::interp_vimg(uk, viTemp, 1.0, uk1);
        LDDMMType::vimg_add_in_place(uk1, viTemp);
        }
      tm_Update.Stop();

      // Dump if requested
      if(flag_dump)
        WriteImageViaCache(uk1.GetPointer(), GetDumpFile(param, "dump_uk1_lev%02d_iter%04d.nii.gz", level, iter));

      // Another layer of smoothing (diffusion-like)
      tm_Gaussian2.Start();
      // LDDMMType::vimg_smooth_withborder(uk1, uk, sigma_post_phys, 1);
      LDDMMType::vimg_smooth(uk1, uk, sigma_post_phys);
      tm_Gaussian2.Stop();

      // Optional incompressibility step
      tm_UpdatePDE.Start();
      if(incompressibility_solver)
        {
        // Compute the divergence of uk.
        LDDMMType::field_divergence(uk, iTemp, true);

        // If using mask, multiply
        if(incompressibility_mask)
          LDDMMType::img_multiply_in_place(iTemp, incompressibility_mask);

        // Dump the divergence before correction
        if(param.flag_dump_moving && 0 == iter % param.dump_frequency)
          {
          char fname[256];
          sprintf(fname, "dump_divv_pre_lev%02d_iter%04d.nii.gz", level, iter);
          LDDMMType::img_write(iTemp, fname);
          }

        // TODO: this should not be temporary!
        typename LDDMMType::ImagePointer pde_soln = LDDMMType::new_img(iTemp);

        // Solve the PDE
        tm_PDE.Start();
        LDDMMType::poisson_pde_zero_boundary_solve(incompressibility_solver, iTemp, pde_soln);
        tm_PDE.Stop();
        if(incompressibility_mask)
          LDDMMType::img_multiply_in_place(pde_soln, incompressibility_mask);

        // Take the gradient of the solution and subtract from uk
        LDDMMType::image_gradient(pde_soln, uk1, true);
        LDDMMType::vimg_subtract_in_place(uk, uk1);

        // Compute the divergence of the updated image. Should be zero
        // Dump the divergence after correction
        if(flag_dump)
          {
          LDDMMType::field_divergence(uk, iTemp, true);
          WriteImageViaCache(iTemp.GetPointer(), GetDumpFile(param, "dump_divv_lev%02d_iter%04d.nii.gz", level, iter));
          }
        }
      tm_UpdatePDE.Stop();
      tm_Iteration.Stop();
      }

    // Store the end result
    uLevel = uk;

    // Compute the jacobian of the deformation field - but only if we iterated at this level
    if(param.iter_per_level[level] > 0)
      {
      LDDMMType::field_jacobian_det(uk, iTemp);
      TReal jac_min, jac_max;
      LDDMMType::img_min_max(iTemp, jac_min, jac_max);
      gout.printf("END OF LEVEL %3d    DetJac Range: %8.4f  to %8.4f \n", level, jac_min, jac_max);

      // Print final metric report
      MultiComponentMetricReport metric_report = this->GetMetricLog()[level].back();
      std::string iter_line = this->PrintIter(level, -1, metric_report);
      gout.printf("%s\n", iter_line.c_str());
      gout.flush();

      // Print timing information
      double n_it = param.iter_per_level[level];
      double t_total = tm_Iteration.GetTotal() / n_it;
      double t_gradient = tm_Gradient.GetTotal() / n_it;
      double t_gaussian = (tm_Gaussian1.GetTotal() + tm_Gaussian2.GetTotal()) / n_it;
      double t_update = (tm_Integration.GetTotal() + tm_Update.GetTotal() + tm_UpdatePDE.GetTotal()) / n_it;
      double t_pde = tm_PDE.GetTotal() / n_it;
      gout.printf("  Avg. Gradient Time        : %6.4fs  %5.2f%% \n", t_gradient, 100 * t_gradient / t_total);
      gout.printf("  Avg. Gaussian Time        : %6.4fs  %5.2f%% \n", t_gaussian, 100 * t_gaussian / t_total);
      if(incompressibility_solver)
        {
        gout.printf("  Avg. PDE Time             : %6.4fs  %5.2f%% \n", t_pde, 100 * t_pde / t_total);
        }
      gout.printf("  Avg. Integration Time     : %6.4fs  %5.2f%% \n", t_update, 100 * t_update / t_total);
      gout.printf("  Avg. Total Iteration Time : %6.4fs \n", t_total);
      }

      // Deallocate the incompressibility solver
      if(incompressibility_solver)
        LDDMMType::poisson_pde_zero_boundary_dealloc(incompressibility_solver);

    }

  // The transformation field is in voxel units. To work with ANTS, it must be mapped
  // into physical offset units - just scaled by the spacing?
  ImageBaseType *warp_ref_space = of_helper.GetReferenceSpace(nlevels - 1);

  if(param.flag_stationary_velocity_mode)
    {
    // Take current warp to 'exponent' power - this is the actual warp
    VectorImagePointer uLevelExp = LDDMMType::new_vimg(uLevel);
    VectorImagePointer uLevelWork = LDDMMType::new_vimg(uLevel);
    LDDMMType::vimg_exp(uLevel, uLevelExp, uLevelWork, param.warp_exponent, 1.0);

    // Write the resulting transformation field (if provided)
    if(param.output.size())
      {
      WriteCompressedWarpInPhysicalSpaceViaCache(warp_ref_space, uLevelExp, param.output.c_str(), param.warp_precision);
      }

    // If asked to write root warp, do so
    if(param.root_warp.size())
      {
      WriteCompressedWarpInPhysicalSpaceViaCache(warp_ref_space, uLevel, param.root_warp.c_str(), 0);
      }

    // Compute the inverse (this is probably unnecessary for small warps)
    if(param.inverse_warp.size())
      {
      // Exponentiate the negative velocity field
      LDDMMType::vimg_exp(uLevel, uLevelExp, uLevelWork, param.warp_exponent, -1.0);
      // of_helper.ComputeDeformationFieldInverse(uLevel, uLevelWork, 0);
      WriteCompressedWarpInPhysicalSpaceViaCache(warp_ref_space, uLevelExp, param.inverse_warp.c_str(), param.warp_precision);
      }
    }
  else
    {
    // Write the resulting transformation field
    WriteCompressedWarpInPhysicalSpaceViaCache(warp_ref_space, uLevel, param.output.c_str(), param.warp_precision);

    // If an inverse is requested, compute the inverse using the Chen 2008 fixed method.
    // A modification of this method is that if convergence is slow, we take the square
    // root of the forward transform.
    //
    // TODO: it would be more efficient to check the Lipschitz condition rather than
    // the brute force approach below
    //
    // TODO: the maximum checks should only be done over the region where the warp is
    // not going outside of the image. Right now, they are meaningless and we are doing
    // extra work when computing the inverse.
    if(param.inverse_warp.size())
      {
      // Compute the inverse
      VectorImagePointer uInverse = LDDMMType::new_vimg(uLevel);
      of_helper.ComputeDeformationFieldInverse(uLevel, uInverse, param.warp_exponent);

      // Write the warp using compressed format
      WriteCompressedWarpInPhysicalSpaceViaCache(warp_ref_space, uInverse, param.inverse_warp.c_str(), param.warp_precision);
      }
    }
  return 0;
}




template <unsigned int VDim, typename TReal>
void GreedyApproach<VDim, TReal>
::WriteCompressedWarpInPhysicalSpaceViaCache(
  ImageBaseType *moving_ref_space, VectorImageType *warp, const char *filename, double precision)
{
  // Define a _float_ output type, even if working with double precision (less space on disk)
  typedef CompressWarpFunctor<VectorImageType, VectorImageType> Functor;

  typedef UnaryPositionBasedFunctorImageFilter<VectorImageType, VectorImageType, Functor> Filter;
  Functor functor(warp, moving_ref_space, precision);

  // Perform the compression
  typename Filter::Pointer filter = Filter::New();
  filter->SetFunctor(functor);
  filter->SetInput(warp);
  filter->Update();

  // Write the resulting image via cache
  WriteImageViaCache(filter->GetOutput(), filename, itk::IOComponentEnum::FLOAT);
}





/**
 * Computes the metric without running any optimization. Metric can be computed
 * at different levels by specifying the iterations array
 */
template <unsigned int VDim, typename TReal>
int GreedyApproach<VDim, TReal>
::ComputeMetric(GreedyParameters &param, MultiComponentMetricReport &metric_report)
{
  // Create an optical flow helper object
  OFHelperType of_helper;

  // Set the scaling factors for multi-resolution
  of_helper.SetDefaultPyramidFactors(1);

  // Set the scaling mode depending on the metric
  if(param.metric == GreedyParameters::MAHALANOBIS)
    of_helper.SetScaleFixedImageWithVoxelSize(true);

  // Read the image pairs to register
  ReadImages(param, of_helper, true);

  // Reference space
  ImageBaseType *refspace = of_helper.GetReferenceSpace(0);

  // Intermediate images
  ImagePointer iTemp = LDDMMType::new_img(refspace);
  VectorImagePointer viTemp = LDDMMType::new_vimg(refspace);
  VectorImagePointer uk = LDDMMType::new_vimg(refspace);
  VectorImagePointer uk1 = LDDMMType::new_vimg(refspace);

  // Load initial transform into uk
  this->LoadInitialTransform(param, of_helper, 0, uk);

  // A pointer to the full warp image - either uk in greedy mode, or uk_exp in diff demons mdoe
  VectorImageType *uFull = uk;
  if(param.flag_stationary_velocity_mode)
    {
    // This is the exponentiation of the stationary velocity field
    // Take current warp to 'exponent' power - this is the actual warp
    VectorImagePointer uk_exp = LDDMMType::new_vimg(refspace);
    LDDMMType::vimg_exp(uk, uk_exp, viTemp, param.warp_exponent, 1.0);
    uFull = uk_exp;
    }

  // Compute the metric
  this->EvaluateMetricForDeformableRegistration(param, of_helper, 0, uFull, metric_report, iTemp, uk1, 1.0);

  // Output the metric image and metric gradient
  if(param.output.length())
    {
    LDDMMType::img_write(iTemp, param.output.c_str());
    }
  if(param.output_metric_gradient.length())
    {
    LDDMMType::vimg_write(uk1, param.output_metric_gradient.c_str());
    }

  return 0;
}


/**
 * This function performs brute force search for similar patches. It generates a discrete displacement
 * field where every pixel in the fixed image is matched to the most similar pixel in the moving image
 * within a certain radius
 */
template <unsigned int VDim, typename TReal>
int GreedyApproach<VDim, TReal>
::RunBrute(GreedyParameters &param)
{
  // Check for valid parameters
  if(param.metric != GreedyParameters::NCC && param.metric != GreedyParameters::WNCC)
    {
    std::cerr << "Brute force search requires NCC metric only" << std::endl;
    return -1;
    }

  if(param.brute_search_radius.size() != VDim)
    {
    std::cerr << "Brute force search radius must be same dimension as the images" << std::endl;
    return -1;
    }

  // Create an optical flow helper object
  OFHelperType of_helper;

  // No multi-resolution
  of_helper.SetDefaultPyramidFactors(1);

  // Read the image pairs to register
  ReadImages(param, of_helper, true);

  // Reference space
  ImageBaseType *refspace = of_helper.GetReferenceSpace(0);

  // Intermediate images
  VectorImagePointer u_best = LDDMMType::new_vimg(refspace);
  VectorImagePointer u_curr = LDDMMType::new_vimg(refspace);
  ImagePointer m_curr = LDDMMType::new_img(refspace);
  ImagePointer m_best = LDDMMType::new_img(refspace);

  // Allocate m_best to a negative value
  m_best->FillBuffer(-100.0);

  // Create a neighborhood for computing offsets
  itk::Neighborhood<float, VDim> dummy_nbr;
  itk::Size<VDim> search_rad = array_caster<VDim>::to_itkSize(param.brute_search_radius);
  itk::Size<VDim> metric_rad = array_caster<VDim>::to_itkSize(param.metric_radius);
  dummy_nbr.SetRadius(search_rad);

  // Iterate over all offsets
  for(unsigned int k = 0; k < dummy_nbr.Size(); k++)
    {
    // Get the offset corresponding to this iteration
    itk::Offset<VDim> offset = dummy_nbr.GetOffset(k);

    // Fill the deformation field with this offset
    typename LDDMMType::Vec vec_offset;
    for(unsigned int i = 0; i < VDim; i++)
      vec_offset[i] = offset[i];
    u_curr->FillBuffer(vec_offset);

    // Perform interpolation and metric computation
    MultiComponentMetricReport metric_report;
    m_curr->FillBuffer(0.0);
    for(unsigned int g = 0; g < of_helper.GetNumberOfInputGroups(); g++)
      of_helper.ComputeNCCMetricAndGradient(g, 0, u_curr, metric_rad, false, m_curr, metric_report);

    // Temp: keep track of number of updates
    unsigned long n_updates = 0;

    // Out of laziness, just take a quick pass over the images
    typename VectorImageType::RegionType rgn = refspace->GetBufferedRegion();
    itk::ImageRegionIterator<VectorImageType> it_u(u_best, rgn);
    itk::ImageRegionConstIterator<ImageType> it_m_curr(m_curr, rgn);
    itk::ImageRegionIterator<ImageType> it_m_best(m_best, rgn);
    for(; !it_m_best.IsAtEnd(); ++it_m_best, ++it_m_curr, ++it_u)
      {
      float v_curr = it_m_curr.Value();
      if(v_curr > it_m_best.Value())
        {
        it_m_best.Set(v_curr);
        it_u.Set(vec_offset);
        ++n_updates;
        }
      }

    std::cout << "offset: " << offset << "     updates: " << n_updates << std::endl;
    }

  LDDMMType::vimg_write(u_best, param.output.c_str());
  LDDMMType::img_write(m_best, "mbest.nii.gz");

  return 0;
}


#include "itkWarpVectorImageFilter.h"
#include "itkWarpImageFilter.h"
#include "itkNearestNeighborInterpolateImageFunction.h"


template <unsigned int VDim, typename TReal>
void GreedyApproach<VDim, TReal>
::MapRASAffineToPhysicalWarp(const vnl_matrix<double> &mat,
                             VectorImagePointer &out_warp)
{
  vnl_matrix<double>  A = mat.extract(VDim, VDim);
  vnl_vector<double> b = mat.get_column(VDim).extract(VDim);

  itk::MultiThreaderBase::Pointer mt = itk::MultiThreaderBase::New();
  mt->ParallelizeImageRegion<VDim>(
        out_warp->GetBufferedRegion(),
        [out_warp,A,b](const itk::ImageRegion<VDim> &region)
    {
    typedef itk::ImageRegionIteratorWithIndex<VectorImageType> IterType;
    vnl_vector<double> q;
    itk::Point<double, VDim> pt, pt2;

    for(IterType it(out_warp, region); !it.IsAtEnd(); ++it)
      {
      // Get the physical position
      // TODO: this calls IsInside() internally, which limits efficiency
      out_warp->TransformIndexToPhysicalPoint(it.GetIndex(), pt);

      // Add the displacement (in DICOM coordinates) and
      for(unsigned int i = 0; i < VDim; i++)
        pt2[i] = pt[i] + it.Value()[i];

      // Switch to NIFTI coordinates
      pt2[0] = -pt2[0]; pt2[1] = -pt2[1];

      // Apply the matrix - get the transformed coordinate in DICOM space
      q = A * pt2.GetVnlVector() + b;
      q[0] = -q[0]; q[1] = -q[1];

      // Compute the difference in DICOM space
      for(unsigned int i = 0; i < VDim; i++)
        it.Value()[i] = q[i] - pt[i];
      }
    }, nullptr);
}

template <unsigned int VDim, typename TReal>
void GreedyApproach<VDim, TReal>
::ReadTransformChain(const std::vector<TransformSpec> &tran_chain,
                     ImageBaseType *ref_space,
                     VectorImagePointer &out_warp,
                     MeshArray *meshes)
{
  // Create the initial transform and set it to zero
  out_warp = VectorImageType::New();
  LDDMMType::alloc_vimg(out_warp, ref_space);

  // Read the sequence of transforms
  for(unsigned int i = 0; i < tran_chain.size(); i++)
    {
    // Read the next parameter
    std::string tran = tran_chain[i].filename;

    // Determine if it's an affine transform
    if(CheckCache<VectorImageType>(tran) || itk::ImageIOFactory::CreateImageIO(tran.c_str(), itk::IOFileModeEnum::ReadMode))
      {
      // Read the next warp
      VectorImagePointer warp_i = ReadImageViaCache<VectorImageType>(tran);

      // Create a temporary warp
      VectorImagePointer warp_tmp = LDDMMType::new_vimg(ref_space);

      // If there is an exponent on the transform spec, handle it
      if(tran_chain[i].exponent != 1)
        {
        // The exponent may be specified as a negative number, in which case we take the negative
        // input and exponentiate it
        double absexp = fabs(tran_chain[i].exponent);
        double n_real = log(absexp) / log(2.0);
        int n = (int) (n_real + 0.5);
        if(fabs(n - n_real) > 1.0e-4)
          throw GreedyException("Currently only power of two exponents are supported for warps");

        // Bring the transform into voxel space
        VectorImagePointer warp_exp = LDDMMType::new_vimg(warp_i);
        VectorImagePointer warp_exp_tmp = LDDMMType::new_vimg(warp_i);
        OFHelperType::PhysicalWarpToVoxelWarp(warp_i, warp_i, warp_i);

        // Square the transform N times (in its own space)
        LDDMMType::vimg_exp(warp_i, warp_exp, warp_exp_tmp, n, tran_chain[i].exponent / absexp);

        // Bring the transform back into physical space
        OFHelperType::VoxelWarpToPhysicalWarp(warp_exp, warp_i, warp_i);
        }

      // Apply the warp to the meshes
      if(meshes)
        for(auto &m : *meshes)
          TransformMeshWarp(m, warp_i);

      // Now we need to compose the current transform and the overall warp.
      LDDMMType::interp_vimg(warp_i, out_warp, 1.0, warp_tmp, false, true);
      LDDMMType::vimg_add_in_place(out_warp, warp_tmp);
      }
    else
      {
      // Read the transform as a matrix
      vnl_matrix<double> mat = ReadAffineMatrixViaCache(tran_chain[i]);

      // Apply the matrix to the meshes
      if(meshes)
        for(auto &m : *meshes)
          TransformMeshAffine(m, mat);

      MapRASAffineToPhysicalWarp(mat, out_warp);
      }
    }
}

#include "itkBinaryThresholdImageFilter.h"
//#include "itkRecursiveGaussianImageFilter.h"
#include "itkSmoothingRecursiveGaussianImageFilter.h"
#include "itkNaryFunctorImageFilter.h"

template <class TInputImage, class TOutputImage>
class NaryLabelVotingFunctor
{
public:
  typedef NaryLabelVotingFunctor<TInputImage,TOutputImage> Self;
  typedef typename TInputImage::PixelType InputPixelType;
  typedef typename TOutputImage::PixelType OutputPixelType;
  typedef std::vector<OutputPixelType> LabelArray;

  NaryLabelVotingFunctor(const LabelArray &labels)
    : m_LabelArray(labels), m_Size(labels.size()) {}

  NaryLabelVotingFunctor() : m_Size(0) {}


  OutputPixelType operator() (const std::vector<InputPixelType> &pix)
  {
    InputPixelType best_val = pix[0];
    int best_index = 0;
    for(int i = 1; i < m_Size; i++)
      if(pix[i] > best_val)
        {
        best_val = pix[i];
        best_index = i;
        }

    return m_LabelArray[best_index];
  }

  bool operator != (const Self &other)
    { return other.m_LabelArray != m_LabelArray; }

protected:
  LabelArray m_LabelArray;
  int m_Size;
};

#include "itkMeshFileReader.h"
#include "itkMeshFileWriter.h"
#include "itkMesh.h"
#include "itkTransformMeshFilter.h"

template <unsigned int VDim, typename TArray>
class PhysicalCoordinateTransform
{
  static void ras_to_lps(const TArray &, TArray &) {}
  static void lps_to_ras(const TArray &, TArray &) {}
};

template <typename TArray>
class PhysicalCoordinateTransform<2, TArray>
{
public:
  static void ras_to_lps(const TArray &src, TArray &trg)
  {
    trg[0] = -src[0];
    trg[1] = -src[1];
  }

  static void lps_to_ras(const TArray &src, TArray &trg)
  {
    trg[0] = -src[0];
    trg[1] = -src[1];
  }
};

template <typename TArray>
class PhysicalCoordinateTransform<3, TArray>
{
public:
  static void ras_to_lps(const TArray &src, TArray &trg)
  {
    trg[0] = -src[0];
    trg[1] = -src[1];
    trg[2] = src[2];
  }

  static void lps_to_ras(const TArray &src, TArray &trg)
  {
    trg[0] = -src[0];
    trg[1] = -src[1];
    trg[2] = src[2];
  }
};

template <typename TArray>
class PhysicalCoordinateTransform<4, TArray>
{
public:
  static void ras_to_lps(const TArray &src, TArray &trg)
  {
    trg[0] = -src[0];
    trg[1] = -src[1];
    trg[2] = src[2];
    trg[3] = src[3];
  }

  static void lps_to_ras(const TArray &src, TArray &trg)
  {
    trg[0] = -src[0];
    trg[1] = -src[1];
    trg[2] = src[2];
    trg[3] = src[3];
  }
};


template <unsigned int VDim, typename TReal>
void GreedyApproach<VDim, TReal>
::TransformMeshAffine(vtkPolyData *mesh, vnl_matrix<double> mat)
{
  vnl_matrix_fixed<double, VDim+1, VDim+1> matfix = mat;
  vnl_vector_fixed<double, VDim+1> x_fix, y_fix; x_fix[VDim] = 1.0;
  for(unsigned int i = 0; i < mesh->GetNumberOfPoints(); i++)
    {
    double *x = mesh->GetPoint(i);
    for(unsigned int d = 0; d < VDim; d++)
      x_fix[d] = x[d];

    y_fix = matfix * x_fix;

    mesh->GetPoints()->SetPoint(i, y_fix.data_block());
    }
}

template <unsigned int VDim, typename TReal>
void GreedyApproach<VDim, TReal>
::TransformMeshWarp(vtkPolyData *mesh, VectorImageType *warp)
{
  typedef FastLinearInterpolator<VectorImageType, TReal, VDim> FastInterpolator;
  typedef itk::Point<TReal, VDim> PointType;
  typedef itk::ContinuousIndex<TReal, VDim> CIndexType;
  FastInterpolator interp(warp);

  // Each vertex is simply multiplied by the matrix
  for(unsigned int i = 0; i < mesh->GetNumberOfPoints(); i++)
    {
    double *x_mesh = mesh->GetPoint(i);

    // Set the initial point
    PointType p_input, p_input_lps, p_out_lps, p_out_ras;
    for(unsigned int d = 0; d < VDim; d++)
      p_input[d] = x_mesh[d];

    // Map the physical coordinate to a continuous index
    PhysicalCoordinateTransform<VDim, PointType>::ras_to_lps(p_input, p_input_lps);

    CIndexType cix;
    typename VectorImageType::PixelType vec;
    vec.Fill(0.0);
    warp->TransformPhysicalPointToContinuousIndex(p_input_lps, cix);
    interp.Interpolate(cix.GetDataPointer(), &vec);

    for(unsigned int d = 0; d < VDim; d++)
      p_out_lps[d] = vec[d] + p_input_lps[d];

    PhysicalCoordinateTransform<VDim, PointType>::lps_to_ras(p_out_lps, p_out_ras);

    mesh->GetPoints()->SetPoint(i, p_out_ras.GetDataPointer());
    }
}


template <unsigned int VDim, typename TReal>
class WarpMeshTransformFunctor : public itk::DataObject
{
public:
  typedef WarpMeshTransformFunctor<VDim, TReal>       Self;
  typedef itk::DataObject                             Superclass;
  typedef itk::SmartPointer<Self>                     Pointer;
  typedef itk::SmartPointer<const Self>               ConstPointer;

  itkTypeMacro(WarpMeshTransformFunctor, itk::DataObject)
  itkNewMacro(Self)

  typedef GreedyApproach<VDim, TReal> GreedyAPI;
  typedef typename GreedyAPI::VectorImageType VectorImageType;
  typedef typename GreedyAPI::ImageBaseType ImageBaseType;
  typedef FastLinearInterpolator<VectorImageType, TReal, VDim> FastInterpolator;
  typedef itk::ContinuousIndex<TReal, VDim> CIndexType;
  typedef itk::Point<TReal, VDim> PointType;

  void SetWarp(VectorImageType *warp)
  {
    if(m_Interpolator) delete m_Interpolator;
    m_Interpolator = new FastInterpolator(warp);
  }

  void SetReferenceSpace(ImageBaseType *ref)
  {
    m_ReferenceSpace = ref;
  }

  PointType TransformPoint(const PointType &x)
  {
    // Our convention is to use NIFTI/RAS coordinates for meshes, whereas ITK
    // uses the DICOM/LPS convention. We transform point to LPS first
    PointType x_lps, phi_x;

    PhysicalCoordinateTransform<VDim, PointType>::ras_to_lps(x, x_lps);

    CIndexType cix;
    typename VectorImageType::PixelType vec;
    vec.Fill(0.0);
    m_ReferenceSpace->TransformPhysicalPointToContinuousIndex(x_lps, cix);
    m_Interpolator->Interpolate(cix.GetDataPointer(), &vec);

    for(unsigned int d = 0; d < VDim; d++)
      {
      phi_x[d] = vec[d] + x_lps[d];
      }


    PhysicalCoordinateTransform<VDim, PointType>::lps_to_ras(phi_x, phi_x);

    return phi_x;
  }

protected:

  WarpMeshTransformFunctor() { m_Interpolator = nullptr; }
  ~WarpMeshTransformFunctor()
  {
    if(m_Interpolator)
      delete m_Interpolator;
  }

private:

  typename ImageBaseType::Pointer m_ReferenceSpace;
  FastInterpolator *m_Interpolator;

};

/**
 * This code computes the jacobian determinant field for a deformation. The
 * recommended mode for this computation is to take the k-th root of the input
 * transformation and then compose the Jacobians
 */
template <unsigned int VDim, typename TReal>
int GreedyApproach<VDim, TReal>
::RunJacobian(GreedyParameters &param)
{
  // Read the warp as a transform chain
  VectorImagePointer warp;

  // Read the warp file
  LDDMMType::vimg_read(param.jacobian_param.in_warp.c_str(), warp);

  // Convert the warp file into voxel units from physical units
  OFHelperType::PhysicalWarpToVoxelWarp(warp, warp, warp);

  // Compute the root of the warp
  VectorImagePointer root_warp = VectorImageType::New();
  LDDMMType::alloc_vimg(root_warp, warp);

  // Allocate a working warp
  VectorImagePointer work_warp = VectorImageType::New();
  LDDMMType::alloc_vimg(work_warp, warp);

  // Compute the root warp, which is not stored in the variable warp
  OFHelperType::ComputeWarpRoot(warp, root_warp, param.warp_exponent);

  // Initialize empty array of Jacobians
  typedef typename LDDMMType::MatrixImageType JacobianImageType;
  typename JacobianImageType::Pointer jac = LDDMMType::new_mimg(warp);

  typename JacobianImageType::Pointer jac_work = LDDMMType::new_mimg(warp);

  // Compute the Jacobian of the root warp
  LDDMMType::field_jacobian(root_warp, jac);

  // Compute the Jacobian matrix of the root warp; jac[a] = D_a (warp)
  for(int k = 0; k < param.warp_exponent; k++)
    {
    // Compute the composition of the Jacobian with itself
    LDDMMType::jacobian_of_composition(jac, jac, root_warp, jac_work);

    // Swap the pointers, so jac points to the actual composed jacobian
    typename JacobianImageType::Pointer temp = jac_work.GetPointer();
    jac_work = jac.GetPointer();
    jac = temp.GetPointer();

    // Compute the composition of the warp with itself, place into root_warp
    LDDMMType::interp_vimg(root_warp, root_warp, 1.0, work_warp);
    LDDMMType::vimg_add_in_place(root_warp, work_warp);
    }

  // At this point, root_warp should hold the original warp, and jac+I will hold
  // the Jacobian of the original warp. We need to compute the determinant
  ImagePointer jac_det = ImageType::New();
  LDDMMType::alloc_img(jac_det, warp);
  LDDMMType::mimg_det(jac, 1.0, jac_det);

  // Write the computed Jacobian
  LDDMMType::img_write(jac_det, param.jacobian_param.out_det_jac.c_str(), itk::IOComponentEnum::FLOAT);
  return 0;
}


template <typename TReal, typename TLabel>
class CompositeToLabelFunctor
{
  public:
    short operator () (itk::VariableLengthVector<TReal> const &p) const { return (short) p[0]; }
};

/**
 * Run the reslice code - simply apply a warp or set of warps to images
 */
template <unsigned int VDim, typename TReal>
int GreedyApproach<VDim, TReal>
::RunReslice(GreedyParameters &param)
{
  // Object for text output
  GreedyStdOut gout(param.verbosity);

  GreedyResliceParameters r_param = param.reslice_param;

  // Check the parameters
  if(!r_param.ref_image.size())
    throw GreedyException("A reference image (-rf) option is required for reslice commands");

  if(r_param.images.size() + r_param.meshes.size() == 0
     && !r_param.out_composed_warp.size()
     && !r_param.out_jacobian_image.size())
    throw GreedyException("No operation specified for reslice mode. "
                          "Use one of -rm, -rs or -rc commands.");

  // Read the fixed as a plain image (we don't care if it's composite)
  typename ImageBaseType::Pointer ref = ReadImageBaseViaCache(r_param.ref_image);

  typedef vtkSmartPointer<vtkPolyData> MeshPointer;
  std::vector<MeshPointer> meshes;
  for(unsigned int i = 0; i < r_param.meshes.size(); i++)
    meshes.push_back(ReadPolyData(r_param.meshes[i].fixed.c_str()));

  // Read the transform chain
  VectorImagePointer warp;
  ReadTransformChain(param.reslice_param.transforms, ref, warp, &meshes);

  // Write the composite warp if requested
  if(r_param.out_composed_warp.size())
    {
    WriteImageViaCache(warp.GetPointer(), r_param.out_composed_warp.c_str(), itk::IOComponentEnum::FLOAT);
    }

  // Compute the Jacobian of the warp if requested
  if(r_param.out_jacobian_image.size())
    {
    ImagePointer iTemp = ImageType::New();
    LDDMMType::alloc_img(iTemp, warp);
    LDDMMType::field_jacobian_det(warp, iTemp);

    WriteImageViaCache(iTemp.GetPointer(), r_param.out_jacobian_image.c_str(), itk::IOComponentEnum::FLOAT);
    }


  // Process image pairs
  for(unsigned int i = 0; i < r_param.images.size(); i++)
    {
    const char *filename = r_param.images[i].moving.c_str();

    // Handle the special case of multi-label images
    if(r_param.images[i].interp.mode == InterpSpec::LABELWISE)
      {
      // The label image is assumed to have a finite set of labels
      typename CompositeImageType::Pointer moving = ReadImageViaCache<CompositeImageType>(filename);
      if(moving->GetNumberOfComponentsPerPixel() > 1)
        throw GreedyException("Label wise interpolation not supported for multi-component images");

      // Cast the image to an image of shorts
      typedef itk::Image<short, VDim> LabelImageType;
      typedef CompositeToLabelFunctor<TReal, short> CastFunctor;
      typedef itk::UnaryFunctorImageFilter<CompositeImageType, LabelImageType, CastFunctor> CastFilter;
      typename CastFilter::Pointer fltCast = CastFilter::New();
      fltCast->SetInput(moving);
      fltCast->Update();
      typename LabelImageType::Pointer label_image = fltCast->GetOutput();

      // Scan the unique labels in the image
      std::set<short> label_set;
      short *labels = label_image->GetBufferPointer();
      int n_pixels = label_image->GetPixelContainer()->Size();

      // Get the list of unique pixels
      short last_pixel = 0;
      for(int j = 0; j < n_pixels; j++)
        {
        short pixel = labels[j];
        if(last_pixel != pixel || i == 0)
          {
          label_set.insert(pixel);
          last_pixel = pixel;
          if(label_set.size() > 1000)
            throw GreedyException("Label wise interpolation not supported for image %s "
                                  "which has over 1000 distinct labels", filename);
          }
        }

      // Turn this set into an array
      std::vector<short> label_array(label_set.begin(), label_set.end());

      // Create a N-way voting filter
      typedef NaryLabelVotingFunctor<ImageType, LabelImageType> VotingFunctor;
      VotingFunctor vf(label_array);

      typedef itk::NaryFunctorImageFilter<ImageType, LabelImageType, VotingFunctor> VotingFilter;
      typename VotingFilter::Pointer fltVoting = VotingFilter::New();
      fltVoting->SetFunctor(vf);

      // Create a mini-pipeline of streaming filters
      for(unsigned int j = 0; j < label_array.size(); j++)
        {
        // Set up a threshold filter for this label
        typedef itk::BinaryThresholdImageFilter<LabelImageType, ImageType> ThresholdFilterType;
        typename ThresholdFilterType::Pointer fltThreshold = ThresholdFilterType::New();
        fltThreshold->SetInput(label_image);
        fltThreshold->SetLowerThreshold(label_array[j]);
        fltThreshold->SetUpperThreshold(label_array[j]);
        fltThreshold->SetInsideValue(1.0);
        fltThreshold->SetOutsideValue(0.0);

        // Set up a smoothing filter for this label
        typedef itk::SmoothingRecursiveGaussianImageFilter<ImageType, ImageType> SmootherType;
        typename SmootherType::Pointer fltSmooth = SmootherType::New();
        fltSmooth->SetInput(fltThreshold->GetOutput());

        // Work out the sigmas for the filter
        if(r_param.images[i].interp.sigma.physical_units)
          {
          fltSmooth->SetSigma(r_param.images[i].interp.sigma.sigma);
          }
        else
          {
          typename SmootherType::SigmaArrayType sigma_array;
          for(unsigned int d = 0; d < VDim; d++)
            sigma_array[d] = r_param.images[i].interp.sigma.sigma * label_image->GetSpacing()[d];
          fltSmooth->SetSigmaArray(sigma_array);
          }

        // TODO: we should really be coercing the output into a vector image to speed up interpolation!
        typedef FastWarpCompositeImageFilter<ImageType, ImageType, VectorImageType> InterpFilter;
        typename InterpFilter::Pointer fltInterp = InterpFilter::New();
        fltInterp->SetMovingImage(fltSmooth->GetOutput());
        fltInterp->SetDeformationField(warp);
        fltInterp->SetUsePhysicalSpace(true);

        fltInterp->Update();

        // Add to the voting filter
        fltVoting->SetInput(j, fltInterp->GetOutput());
        }

      // TODO: test out streaming!
      // Run this big pipeline
      fltVoting->Update();

      // Save
      WriteImageViaCache(fltVoting->GetOutput(), r_param.images[i].output.c_str());
      }
    else
      {
      // Read the input image and record its type
      itk::IOComponentEnum comp;
      CompositeImagePointer moving = ReadImageViaCache<CompositeImageType>(filename, &comp);

      // Allocate the warped image
      CompositeImagePointer warped = LDDMMType::new_cimg(ref, moving->GetNumberOfComponentsPerPixel());

      // Perform the warp
      LDDMMType::interp_cimg(moving, warp, warped,
                             r_param.images[i].interp.mode == InterpSpec::NEAREST,
                             true, r_param.images[i].interp.outside_value);

      // Write, casting to the input component type
      WriteImageViaCache(warped.GetPointer(), r_param.images[i].output.c_str(), comp);
      }
    }

  // Save the meshes
  for(unsigned int i = 0; i < r_param.meshes.size(); i++)
    WritePolyData(meshes[i], r_param.meshes[i].output.c_str());


  // Process meshes
  /*
  for(unsigned int i = 0; i < r_param.meshes.size(); i++)
    {
    typedef itk::Mesh<TReal, VDim> MeshType;
    typedef itk::MeshFileReader<MeshType> MeshReader;
    typename MeshType::Pointer mesh;

    if(itksys::SystemTools::GetFilenameExtension(r_param.meshes[i].fixed) == ".csv")
      {
      mesh = MeshType::New();

      std::ifstream fin(r_param.meshes[i].fixed.c_str());
      std::string f_line, f_token;
      unsigned int n_pts = 0;
      while(std::getline(fin, f_line))
        {
        std::istringstream iss(f_line);
        itk::Point<TReal, VDim> pt;
        for(unsigned int a = 0; a < VDim; a++)
          {
          if(!std::getline(iss, f_token, ','))
            throw GreedyException("Error reading CSV file, line %s", f_line.c_str());
          pt[a] = atof(f_token.c_str());
          }
        mesh->SetPoint(n_pts++, pt);
        }
      }
    else
      {
      typename MeshReader::Pointer reader = MeshReader::New();
      reader->SetFileName(r_param.meshes[i].fixed.c_str());
      reader->Update();
      mesh = reader->GetOutput();
      }

    typedef WarpMeshTransformFunctor<VDim, TReal> TransformType;
    typename TransformType::Pointer transform = TransformType::New();
    transform->SetWarp(warp);
    transform->SetReferenceSpace(ref);

    typedef itk::TransformMeshFilter<MeshType, MeshType, TransformType> FilterType;
    typename FilterType::Pointer filter = FilterType::New();
    filter->SetTransform(transform);
    filter->SetInput(mesh);
    filter->Update();

    mesh = filter->GetOutput();

    if(itksys::SystemTools::GetFilenameExtension(r_param.meshes[i].output) == ".csv")
      {
      std::ofstream out(r_param.meshes[i].output.c_str());
      for(unsigned int i = 0; i < mesh->GetNumberOfPoints(); i++)
        {
        itk::Point<TReal, VDim> pt = mesh->GetPoint(i);
        for(unsigned int a = 0; a < VDim; a++)
          out << pt[a] << (a < VDim-1 ? "," : "\n");
        }
      }
    else
      {
      typedef itk::MeshFileWriter<MeshType> MeshWriter;
      typename MeshWriter::Pointer writer = MeshWriter::New();
      writer->SetInput(mesh);
      writer->SetFileName(r_param.meshes[i].output.c_str());
      writer->Update();
      }
    }
    */



  return 0;
}

template <unsigned int VDim, typename TReal>
void GreedyApproach<VDim, TReal>
::ComputeImageMoments(CompositeImageType *image,
                      const vnl_vector<float> &weights,
                      VecFx &m1, MatFx &m2)
{
  int n = image->GetNumberOfComponentsPerPixel();
  TReal sum_I = 0.0;
  m1.fill(0.0); m2.fill(0.0);

  typedef itk::ImageRegionConstIteratorWithIndex<CompositeImageType> Iterator;
  for(Iterator it(image, image->GetBufferedRegion()); !it.IsAtEnd(); ++it)
    {
    typedef itk::Point<TReal, VDim> PointType;
    PointType p_lps, p_ras;
    image->TransformIndexToPhysicalPoint(it.GetIndex(), p_lps);
    PhysicalCoordinateTransform<VDim, PointType>::lps_to_ras(p_lps, p_ras);
    VecFx X(p_ras.GetDataPointer());
    MatFx X2 = outer_product(X, X);

    typename CompositeImageType::PixelType pix = it.Get();

    // Just weight the components of intensity by weight vector - this sort of makes sense?
    TReal val = 0.0;
    for(int k = 0; k < n; k++)
      val += weights[k] * pix[k];

    sum_I += val;
    m1 += X * val;
    m2 += X2 * val;
    }

  // Compute the mean and covariance from the sum of squares
  m1 = m1 / sum_I;
  m2 = (m2 - sum_I *  outer_product(m1, m1)) / sum_I;
}

template <unsigned int VDim, typename TReal>
int GreedyApproach<VDim, TReal>
::RunAlignMoments(GreedyParameters &param)
{
  typedef PhysicalSpaceAffineCostFunction<VDim, TReal> PhysicalSpaceAffineCostFunction;

  // Create an optical flow helper object
  OFHelperType of_helper;

  // No multi-resolution
  of_helper.SetDefaultPyramidFactors(1);

  // Read the image pairs to register
  ReadImages(param, of_helper, false);

  // We do not support multiple groups for now
  if(of_helper.GetNumberOfInputGroups() != 1)
    throw GreedyException("Multiple input groups not supported in matching by moments");

  // Compute the moments of intertia for the fixed and moving images. For now
  // this is done in an iterator loop, out of laziness. Should be converted to
  // a filter if this whole moments business proves useful
  VecFx m1f, m1m;
  MatFx m2f, m2m;


  std::cout << "--- MATCHING BY MOMENTS OF ORDER " << param.moments_order << " ---" << std::endl;

  ComputeImageMoments(of_helper.GetFixedComposite(0,0), of_helper.GetWeights(0), m1f, m2f);

  std::cout << "Fixed Mean        : " << m1f << std::endl;
  std::cout << "Fixed Covariance  : " << std::endl << m2f << std::endl;

  ComputeImageMoments(of_helper.GetMovingComposite(0,0), of_helper.GetWeights(0), m1m, m2m);

  std::cout << "Moving Mean       : " << m1m << std::endl;
  std::cout << "Moving Covariance : " << std::endl << m2m << std::endl;

  // This flag forces no rotation, only flip
  if(param.moments_order == 1 || param.flag_moments_id_covariance)
    {
    m2f.set_identity();
    m2m.set_identity();
    }

  // Decompose covariance matrices into eigenvectors and eigenvalues
  vnl_vector<TReal> Df, Dm;
  vnl_matrix<TReal> Vf, Vm;
  vnl_symmetric_eigensystem_compute<TReal>(m2f.as_matrix(), Vf, Df);
  vnl_symmetric_eigensystem_compute<TReal>(m2m.as_matrix(), Vm, Dm);

  // Create a rigid registration problem
  PhysicalSpaceAffineCostFunction cost_fn(&param, this, 0, 0, &of_helper);

  // The best set of coefficients and the associated match value
  vnl_vector<double> xBest;
  TReal xBestMatch = vnl_numeric_traits<TReal>::maxval;

  // Generate all possible flip matrices
  int n_flip = 1 << VDim;
  for(int k_flip = 0; k_flip < n_flip; k_flip++)
    {
    // If using first moments, ignore all flips, only allow identity
    if(param.moments_order == 1 && k_flip != n_flip - 1)
      continue;

    // Generate the flip matrix
    MatFx F(0.0);
    for(unsigned int d = 0; d < VDim; d++)
      F(d,d) = (k_flip & (1 << d)) ? 1 : -1;;

    // Compute the rotation matrix - takes fixed coordinates into moving space
    MatFx R = Vm * F * Vf.transpose();
    VecFx b = m1m - R * m1f;

    vnl_matrix<TReal> A(VDim+1, VDim+1, 0.0);
    A.set_identity();
    A.update(R.as_matrix(), 0, 0);
    for(unsigned int d= 0 ;d< VDim;d++)
      A(d,VDim) = b[d];

    // Ignore flips with the wrong determinant
    double det_R = vnl_determinant(R);
    if((param.moments_order == 2 && param.moments_flip_determinant == 1 && det_R < 0) ||
       (param.moments_order == 2 && param.moments_flip_determinant == -1 && det_R > 0))
      {
      continue;
      }

    // Generate affine coefficients from the rotation and shift
    vnl_vector<double> x(cost_fn.get_number_of_unknowns());
    flatten_affine_transform(R, b, x.data_block());

    // Compute similarity
    double f = 0.0;
    cost_fn.compute(x, &f, NULL);

    std::cout << "Metric for flip " << F.get_diagonal() << " : " << f << std::endl;

    // Compare
    if(xBestMatch > f || xBest.size() == 0)
      {
      xBestMatch = f;
      xBest = x;
      }
    }

  // Save the best transform
  typename LinearTransformType::Pointer tran = LinearTransformType::New();
  cost_fn.GetTransform(xBest, tran);
  vnl_matrix<double> Q_physical = MapAffineToPhysicalRASSpace(of_helper, 0, 0, tran);
  this->WriteAffineMatrixViaCache(param.output, Q_physical);

  return 0;
}

/**
 * Post-hoc warp inversion - the Achilles heel of non-symmetric registration :(
 */
template <unsigned int VDim, typename TReal>
int GreedyApproach<VDim, TReal>
::RunInvertWarp(GreedyParameters &param)
{
  // Read the warp as a transform chain
  VectorImagePointer warp;

  // Read the warp file
  LDDMMType::vimg_read(param.invwarp_param.in_warp.c_str(), warp);

  // Convert the warp file into voxel units from physical units
  OFHelperType::PhysicalWarpToVoxelWarp(warp, warp, warp);


  // Compute the inverse of the warp
  VectorImagePointer uInverse = VectorImageType::New();
  LDDMMType::alloc_vimg(uInverse, warp);
  OFHelperType::ComputeDeformationFieldInverse(warp, uInverse, param.warp_exponent, true);

  // Write the warp using compressed format
  WriteCompressedWarpInPhysicalSpaceViaCache(warp, uInverse, param.invwarp_param.out_warp.c_str(), param.warp_precision);

  return 0;
}

/**
 * Post-hoc warp root
 */
template <unsigned int VDim, typename TReal>
int GreedyApproach<VDim, TReal>
::RunRootWarp(GreedyParameters &param)
{
  // Read the warp as a transform chain
  VectorImagePointer warp;

  // Read the warp file
  LDDMMType::vimg_read(param.warproot_param.in_warp.c_str(), warp);

  // Convert the warp file into voxel units from physical units
  OFHelperType::PhysicalWarpToVoxelWarp(warp, warp, warp);

  // Allocate the root
  VectorImagePointer warp_root = VectorImageType::New();
  LDDMMType::alloc_vimg(warp_root, warp);

  // Take the n-th root
  OFHelperType::ComputeWarpRoot(warp, warp_root, param.warp_exponent, 1e-6);

  // Write the warp using compressed format
  WriteCompressedWarpInPhysicalSpaceViaCache(warp, warp_root, param.warproot_param.out_warp.c_str(), param.warp_precision);

  return 0;
}

template <unsigned int VDim, typename TReal>
int GreedyApproach<VDim, TReal>
::RunMetric(GreedyParameters &param)
{
  MultiComponentMetricReport metric_report;
  this->ComputeMetric(param, metric_report);

  printf("Metric Report:\n");
  for (unsigned i = 0; i < metric_report.ComponentPerPixelMetrics.size(); i++)
    printf("  Component %d: %8.6f", i, metric_report.ComponentPerPixelMetrics[i]);
  printf("  Total = %8.6f\n", metric_report.TotalPerPixelMetric);

  return 0;
}

class LinearIntensityMapping
{
public:
  typedef LinearIntensityMapping Self;

  double operator() (double g) const
    { return MapInternalToNative(g); }

  double MapInternalToNative(double internal) const
    { return internal * scale + shift; }

  double MapNativeToInternal(double native) const
    { return (native - shift) / scale; }

  LinearIntensityMapping() : scale(1.0), shift(0.0) {}
  LinearIntensityMapping(double a, double b) : scale(a), shift(b) {}

  bool operator != (const Self &other) const
    { return scale != other.scale || shift != other.shift; }

protected:
  double scale;
  double shift;
};

class IdentityIntensityMapping
{
public:
  typedef IdentityIntensityMapping Self;

  double operator() (double g) const
    { return g; }

  double MapInternalToNative(double internal) const
    { return internal; }

  double MapNativeToInternal(double native) const
    { return native; }

  bool operator != (const Self &other) const { return false; }
};


template <class TInputImage, class TOutputImage, class TFunctor>
class UnaryFunctorImageToSingleComponentVectorImageFilter
    : public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  typedef UnaryFunctorImageToSingleComponentVectorImageFilter<TInputImage, TOutputImage, TFunctor> Self;
  typedef itk::ImageToImageFilter<TInputImage, TOutputImage> Superclass;
  typedef itk::SmartPointer<Self> Pointer;
  typedef itk::SmartPointer< const Self >  ConstPointer;

  typedef TInputImage InputImageType;
  typedef TOutputImage OutputImageType;
  typedef TFunctor FunctorType;

  typedef typename Superclass::OutputImageRegionType OutputImageRegionType;

  /** Run-time type information (and related methods). */
  itkTypeMacro(UnaryFunctorImageToSingleComponentVectorImageFilter, ImageToImageFilter)
  itkNewMacro(Self)

  /** ImageDimension constants */
  itkStaticConstMacro(InputImageDimension, unsigned int,
                      TInputImage::ImageDimension);
  itkStaticConstMacro(OutputImageDimension, unsigned int,
                      TOutputImage::ImageDimension);

  void SetFunctor(const FunctorType &functor)
  {
    if(m_Functor != functor)
      {
      m_Functor = functor;
      this->Modified();
      }
  }

  itkGetConstReferenceMacro(Functor, FunctorType)

  void DynamicThreadedGenerateData(const OutputImageRegionType & outputRegionForThread) ITK_OVERRIDE;


protected:

  UnaryFunctorImageToSingleComponentVectorImageFilter() {}
  virtual ~UnaryFunctorImageToSingleComponentVectorImageFilter() {}

  FunctorType m_Functor;

};

template <class TInputImage, class TOutputImage, class TFunctor>
void
UnaryFunctorImageToSingleComponentVectorImageFilter<TInputImage, TOutputImage, TFunctor>
::DynamicThreadedGenerateData(const OutputImageRegionType &outputRegionForThread)
{
  // Use our fast iterators for vector images
  typedef itk::ImageLinearIteratorWithIndex<OutputImageType> IterBase;
  typedef IteratorExtender<IterBase> IterType;

  typedef typename OutputImageType::InternalPixelType OutputComponentType;
  typedef typename InputImageType::InternalPixelType InputComponentType;

  // Define the iterators
  IterType outputIt(this->GetOutput(), outputRegionForThread);
  int line_len = outputRegionForThread.GetSize(0);

  // Using a generic ITK iterator for the input because it supports RLE images and adaptors
  itk::ImageScanlineConstIterator< InputImageType > inputIt(this->GetInput(), outputRegionForThread);

  while ( !inputIt.IsAtEnd() )
    {
    // Get the pointer to the input and output pixel lines
    OutputComponentType *out = outputIt.GetPixelPointer(this->GetOutput());

    for(int i = 0; i < line_len; i++, ++inputIt)
      {
      out[i] = m_Functor(inputIt.Get());
      }

    outputIt.NextLine();
    inputIt.NextLine();
    }
}

template <class TInputImage, class TIntensityMapping, typename TReal>
static typename itk::VectorImage<TReal, 3>::Pointer
CastToVectorImage (TInputImage *input)
{
  typedef UnaryFunctorImageToSingleComponentVectorImageFilter<
      TInputImage, itk::VectorImage<TReal, 3>, TIntensityMapping> FilterType;
  typedef itk::ImageSource<itk::VectorImage<TReal, 3>> VectorImageSource;

  TIntensityMapping intensityMapping;
  itk::SmartPointer<FilterType> filter = FilterType::New();
  filter->SetInput(input);
  filter->SetFunctor(intensityMapping);
  itk::SmartPointer<VectorImageSource> imgSource = filter.GetPointer();
  imgSource->UpdateOutputInformation();
  imgSource->Update();
  return imgSource->GetOutput();
}

// Interpolation method for resampleing
enum InterpolationMode { Linear=0, NearestNeighbor };

template <typename TImageType>
itk::SmartPointer<TImageType>
Resample3DImage(TImageType* input, double factor,
                  InterpolationMode intpMode, double smooth_stdev = 0)
{
  typedef itk::DiscreteGaussianImageFilter<TImageType,TImageType> SmoothFilter;
  typename TImageType::Pointer imageToResample = input;

  // Smooth image if needed
  if (smooth_stdev > 0)
    {
    typename SmoothFilter::Pointer fltDSSmooth = SmoothFilter::New();
    typename SmoothFilter::ArrayType variance;
    for (int i = 0; i < 3; ++i)
      variance[i] = smooth_stdev * smooth_stdev;

    fltDSSmooth->SetInput(input);
    fltDSSmooth->SetVariance(variance);
    fltDSSmooth->UseImageSpacingOn();
    fltDSSmooth->Update();
    imageToResample = fltDSSmooth->GetOutput();
    }

  // Create resampled images
  typedef itk::ResampleImageFilter<TImageType, TImageType> ResampleFilter;
  typedef itk::LinearInterpolateImageFunction<TImageType, double> LinearInterpolator;
  typedef itk::NearestNeighborInterpolateImageFunction<TImageType, double> NNInterpolator;

  typename ResampleFilter::Pointer fltResample = ResampleFilter::New();
  fltResample->SetInput(imageToResample);
  fltResample->SetTransform(itk::IdentityTransform<double, 3u>::New());

  switch (intpMode)
    {
    case InterpolationMode::Linear:
      fltResample->SetInterpolator(LinearInterpolator::New());
      break;
    case InterpolationMode::NearestNeighbor:
      fltResample->SetInterpolator(NNInterpolator::New());
      break;
    default:
      throw GreedyException("Unkown Interpolation Mode");
    }

  typename TImageType::SizeType sz;
  for(size_t i = 0; i < 3; i++)
    sz[i] = (unsigned long)(imageToResample->GetBufferedRegion().GetSize(i) * factor + 0.5);

  // Compute the spacing of the new image
  typename TImageType::SpacingType spc_pre = imageToResample->GetSpacing();
  typename TImageType::SpacingType spc_post = spc_pre;
  for(size_t i = 0; i < 3; i++)
    spc_post[i] *= imageToResample->GetBufferedRegion().GetSize()[i] * 1.0 / sz[i];

  // Get the bounding box of the input image
  typename TImageType::PointType origin_pre = imageToResample->GetOrigin();

  // Recalculate the origin. The origin describes the center of voxel 0,0,0
  // so that as the voxel size changes, the origin will change as well.
  typename TImageType::SpacingType off_pre = (imageToResample->GetDirection() * spc_pre) * 0.5;
  typename TImageType::SpacingType off_post = (imageToResample->GetDirection() * spc_post) * 0.5;
  typename TImageType::PointType origin_post = origin_pre - off_pre + off_post;

  // Set the image sizes and spacing.
  fltResample->SetSize(sz);
  fltResample->SetOutputSpacing(spc_post);
  fltResample->SetOutputOrigin(origin_post);
  fltResample->SetOutputDirection(imageToResample->GetDirection());

  // Set the unknown intensity to positive value
  fltResample->SetDefaultPixelValue(0);

  // Perform resampling
  fltResample->UpdateLargestPossibleRegion();

  return fltResample->GetOutput();
}

template <unsigned int VDim, typename TReal>
int GreedyApproach<VDim, TReal>
::RunPropagation(GreedyParameters &param)
{
  // Variable stores working data for propagation
  PropagationData<TReal> pData;

  GreedyPropagationParameters prop_param = param.propagation_param;
  const unsigned int refTP = prop_param.refTP;

  // ------------------------------------------
  //   Preparation
  // ------------------------------------------

  // Read Image 4D
  pData.img4d = ReadImageViaCache<Image4DType>(param.propagation_param.img4d);

  typename Image4DType::DirectionType img4d_dir = pData.img4d->GetDirection();

  auto nt = pData.img4d->GetBufferedRegion().GetSize()[3];

  // Validate reference tp and target tps
  if (refTP > nt)
    throw GreedyException("Reference time point %d is greater than total number of time points %d",
                          refTP, nt);

  pData.tp_data[refTP].img = ExtractTimePointImage(pData.img4d, refTP);
  pData.tp_data[refTP].img_srs = Resample3DImage<Image3DType>(
        pData.tp_data[refTP].img,0.5, InterpolationMode::Linear, 1);

  pData.tp_data[refTP].img_srs->Print(std::cout);

  // Extract 3D Images
  for (size_t tp : prop_param.targetTPs)
    {
    TimePointData<TReal> tpData;

    if (tp > nt)
      throw GreedyException("Target time point %d is greater than total number of time points %d",
                            tp, nt);

    tpData.img = ExtractTimePointImage(pData.img4d, tp);
    tpData.img_srs = Resample3DImage<Image3DType>(
          tpData.img, 0.5, InterpolationMode::Linear, 1);

    pData.tp_data[tp] = tpData;
    }

  typename Image3DType::DirectionType img_direction, seg_direction;
  img_direction = pData.tp_data[refTP].img->GetDirection();

  // Read Segmentation pairs
  for (auto it : param.propagation_param.segpair)
    {
    std::cout << "Processing reference segmentation: " << it.refseg << std::endl;

    PropagationSegGroup<TReal> segGroup;
    segGroup.seg_ref = ReadImageViaCache<LabelImageType>(it.refseg);
    segGroup.outdir = it.outsegdir;

    seg_direction = segGroup.seg_ref->GetDirection();

    if (img_direction != seg_direction)
      {
      throw GreedyException("Image and Segmentation orientations do not match. Segmentation file %s",
                            it.refseg.c_str());
      }

    // Thresholding
    typedef itk::BinaryThresholdImageFilter<LabelImageType, LabelImageType> ThresholdFilter;
    typename ThresholdFilter::Pointer fltThreshold = ThresholdFilter::New();
    fltThreshold->SetInput(segGroup.seg_ref);
    fltThreshold->SetLowerThreshold(1);
    fltThreshold->SetUpperThreshold(SHRT_MAX); // +inf
    fltThreshold->SetInsideValue(1);
    fltThreshold->SetOutsideValue(0);
    fltThreshold->Update();

    // Label dilation
    typedef itk::BinaryBallStructuringElement<TReal, 3u> Element;
    typename Element::SizeType sz = { 10, 10, 10 };
    Element elt;
    elt.SetRadius(sz);
    elt.CreateStructuringElement();

    typedef itk::BinaryDilateImageFilter<LabelImageType, LabelImageType, Element> DilateFilter;
    typename DilateFilter::Pointer fltDilation = DilateFilter::New();
    fltDilation->SetInput(fltThreshold->GetOutput());
    fltDilation->SetDilateValue(1);
    fltDilation->SetKernel(elt);
    fltDilation->Update();

    // Resampling
    segGroup.seg_ref_srs =
        Resample3DImage<LabelImageType>(
          fltDilation->GetOutput(), 0.5, InterpolationMode::NearestNeighbor);

    pData.seg_list.push_back(segGroup);
    }

  // We only use the first reference seg to run registration
  // -- The rest of the seg inputs only participate in warping
  pData.tp_data[refTP].seg = pData.seg_list[0].seg_ref;
  pData.tp_data[refTP].seg_srs = pData.seg_list[0].seg_ref_srs;

  /*
  // Debug: write out extracted tp images
  for (auto it : pData.seg_list)
    {
    auto dbgout = it.outdir + '/' + "seg_ref_srs.nii.gz";
    WriteImageViaCache<Image3DType>(it.seg_ref_srs, dbgout);
    }
  */

  // Create a tp list for forward propagation chain
  std::vector<unsigned int> forward_tps { refTP };
  std::vector<unsigned int> backward_tps { refTP };

  for (unsigned int tp : prop_param.targetTPs)
    {
    if (tp < refTP )
      backward_tps.push_back(tp);
    else if (tp > refTP)
      forward_tps.push_back(tp);
    }

  std::sort(forward_tps.begin(), forward_tps.end());
  std::sort(backward_tps.rbegin(), backward_tps.rend());

  std::cout << "forward tps: ";
  for (auto tp : forward_tps)
    std::cout << " " << tp;
  std::cout << std::endl;
  std::cout << "backward tps: ";
  for (auto tp : backward_tps)
    std::cout << " " << tp;
  std::cout << std::endl << std::flush;

  // ------------------------------------------
  //   Forward and Backward Propagation
  // ------------------------------------------


  for (size_t i = 1; i < forward_tps.size(); ++i)
    {
    const auto crntTP = forward_tps[i];
    const auto prevTP = forward_tps[i - 1];
    // Run affine to get transformation for reslicing
    RunPropagationAffine(param, pData, prevTP, crntTP);

    // Run deformable to get warp image for reslicing
    RunPropagationDeformable(param, pData, prevTP, crntTP);

    // Set the transformation chain
    // -- add all <affine, -1> <deform> from reference TP to current TP
    std::cout << "Processing transform chain..." << std::endl;
    for (size_t j = 1; j <= i; ++j)
      {
      const auto ct = forward_tps[j];
      const auto pt = forward_tps[j - 1];

      std::cout << "-- Adding transform for tp=" << ct << std::endl;

      auto &crnt_data = pData.tp_data[ct];
      auto &prev_data = pData.tp_data[pt];
      auto affine = crnt_data.affine_to_prev;
      auto deform = crnt_data.deform_from_prev;
      // Copy previous transform specs as a starting point
      crnt_data.transform_specs = prev_data.transform_specs;
      TimePointTransformSpec<TReal> spec(affine, deform);
      crnt_data.transform_specs.push_back(spec);
      }

    // Run reslicing to warp seg_ref_srs to currrent time point
    RunPropagationReslice(param, pData, prevTP, crntTP);
    }


  // ------------------------------------------
  //   Full Resolution Propagation
  // ------------------------------------------

  return 0;
}





template <unsigned int VDim, typename TReal>
void
GreedyApproach<VDim, TReal>
::RunPropagationAffine(GreedyParameters &glparam, PropagationData<TReal> &pData
                                   ,unsigned int tp_prev, unsigned int tp_crnt)
{
  /*
   * greedy -d 3 -a -ia-identity -dof 6 -s 3mm 1.5mm -i {Dm} {Dn} -gm {DSm} -o [A:n-m]
   */

  std::cout << "[RunPropaAffine] Started prev=" << tp_prev << "; crnt="
            << tp_crnt << std::endl;

  typedef TimePointData<TReal> TPDataType;

  // Get relevant tp data
  TPDataType &tpdata_prev = pData.tp_data[tp_prev];
  TPDataType &tpdata_crnt = pData.tp_data[tp_crnt];

  // Set greedy parameters
  GreedyParameters param;
  GreedyApproach<3u, TReal> *GreedyAPI = new GreedyApproach<3u, TReal>();

  // Set input images
  GreedyInputGroup ig;
  ImagePairSpec ip;
  ip.weight = 1.0;
  ip.fixed = "img_fixed";
  ip.moving = "img_moving";
  ig.inputs.push_back(ip);

  auto casted_fix = CastToVectorImage<Image3DType, LinearIntensityMapping, TReal>(
        tpdata_prev.img_srs);
  auto casted_mov = CastToVectorImage<Image3DType, LinearIntensityMapping, TReal>(
        tpdata_crnt.img_srs);

  GreedyAPI->AddCachedInputObject(ip.fixed, casted_fix);
  GreedyAPI->AddCachedInputObject(ip.moving, casted_mov);

  // Set mask images
  ig.fixed_mask = "mask_fixed";
  auto casted_mask = TimePointData<TReal>::CastLabelToDoubleImage(tpdata_prev.seg_srs);
  GreedyAPI->AddCachedInputObject(ig.fixed_mask, casted_mask);

  param.metric = glparam.metric;
  param.metric_radius = glparam.metric_radius;
  param.affine_dof = GreedyParameters::DOF_RIGID;
  param.iter_per_level = glparam.iter_per_level;
  param.affine_init_mode = AffineInitMode::RAS_IDENTITY;

  // Check smoothing parameters. If greedy default detected, change to propagation default.
  SmoothingParameters default_pre = { 1.7320508076, false }, prop_default_pre = { 3.0, true };
  SmoothingParameters default_post = { 0.7071067812, false }, prop_default_post = { 1.5, true };
  param.sigma_pre = (glparam.sigma_pre == default_pre) ? prop_default_pre : glparam.sigma_pre;
  param.sigma_post = (glparam.sigma_post == default_post) ? prop_default_post : glparam.sigma_post;

  // Add the input group to the parameters
  //  GreedyParameters may create an empty input group during contruction
  //  Remove the defaultly constructed group, replace it with ig
  if (param.input_groups.size() > 0)
    param.input_groups.clear();

  param.input_groups.push_back(ig);

  // Configure output
  param.output = "affine_to_prev";
  GreedyAPI->AddCachedOutputObject(param.output, tpdata_crnt.affine_to_prev);

  int ret = GreedyAPI->RunAffine(param);

  delete GreedyAPI;

  std::cout << "-- Affine Completed. ret=" << ret << std::endl;
  //tpdata_crnt.affine_to_prev.Print(std::cout);
}

template <unsigned int VDim, typename TReal>
void
GreedyApproach<VDim, TReal>
::RunPropagationDeformable(GreedyParameters &glparam, PropagationData<TReal> &pData
                                   ,unsigned int tp_prev, unsigned int tp_crnt)
{
  /*
   * greedy -d 3 -m <MET> -n <MRS> -threads <THR>
   * -s 3mm 1.5mm -it [A:n-m]
   * -i {Dm} {Dn}
   * -gm {DSm}
   * -o [D:n-m] -oinv [D:m-n]
  */

  std::cout << "[RunPropaDeformable] Started prev=" << tp_prev << "; crnt="
            << tp_crnt << std::endl;

  typedef TimePointData<TReal> TPDataType;

  // Get relevant tp data
  TPDataType &tpdata_prev = pData.tp_data[tp_prev];
  TPDataType &tpdata_crnt = pData.tp_data[tp_crnt];

  // Set greedy parameters
  GreedyParameters param;
  GreedyApproach<3u, TReal> *GreedyAPI = new GreedyApproach<3u, TReal>();

  // Set input images
  GreedyInputGroup ig;
  ImagePairSpec ip;
  ip.weight = 1.0;
  ip.fixed = "img_fixed";
  ip.moving = "img_moving";
  ig.inputs.push_back(ip);

  auto casted_fix = CastToVectorImage<Image3DType, LinearIntensityMapping, TReal>(
        tpdata_prev.img_srs);
  auto casted_mov = CastToVectorImage<Image3DType, LinearIntensityMapping, TReal>(
        tpdata_crnt.img_srs);

  GreedyAPI->AddCachedInputObject(ip.fixed, casted_fix);
  GreedyAPI->AddCachedInputObject(ip.moving, casted_mov );

  // Set mask images
  ig.fixed_mask = "mask_fixed";
  auto casted_mask = TimePointData<TReal>::CastLabelToDoubleImage(tpdata_prev.seg_srs);
  GreedyAPI->AddCachedInputObject(ig.fixed_mask, casted_mask);

  param.metric = glparam.metric;
  param.metric_radius = glparam.metric_radius;
  param.iter_per_level = glparam.iter_per_level;

  // Check smoothing parameters. If greedy default detected, change to propagation default.
  SmoothingParameters default_pre = { 1.7320508076, false }, prop_default_pre = { 3.0, true };
  SmoothingParameters default_post = { 0.7071067812, false }, prop_default_post = { 1.5, true };
  param.sigma_pre = (glparam.sigma_pre == default_pre) ? prop_default_pre : glparam.sigma_pre;
  param.sigma_post = (glparam.sigma_post == default_post) ? prop_default_post : glparam.sigma_post;

  // Add the input group to the parameters
  //  GreedyParameters may create an empty input group during contruction
  //  Remove the defaultly constructed group, replace it with ig
  if (param.input_groups.size() > 0)
    param.input_groups.clear();

  param.input_groups.push_back(ig);

  // Configure output
  typedef LDDMMData<TReal, 3> LDDMM3DType;

  param.output = "deform_to_prev";
  tpdata_crnt.deform_to_prev = LDDMM3DType::new_vimg(tpdata_crnt.img_srs);
  GreedyAPI->AddCachedOutputObject(param.output, tpdata_crnt.deform_to_prev);


  param.inverse_warp = "deform_from_prev";
  tpdata_crnt.deform_from_prev = LDDMM3DType::new_vimg(tpdata_crnt.img_srs);
  GreedyAPI->AddCachedOutputObject(param.inverse_warp, tpdata_crnt.deform_from_prev);

  int ret = GreedyAPI->RunDeformable(param);

  std::cout << "-- Deformable Completed: ret=" << ret << std::endl;
  delete GreedyAPI;

}


template <unsigned int VDim, typename TReal>
void
GreedyApproach<VDim, TReal>
::RunPropagationReslice(GreedyParameters &glparam, PropagationData<TReal> &pData
                                   ,unsigned int tp_prev, unsigned int tp_crnt)
{
  /*
   * greedy -d 3 -ri NN -rf {Dn} -r [W:r-n] -rm {DSr} {DSn}
  */

  std::cout << "[RunPropaReslice] Started prev=" << tp_prev << "; crnt="
            << tp_crnt << std::endl;

  typedef TimePointData<TReal> TPDataType;

  // Get relevant tp data
  TPDataType &tpdata_ref = pData.tp_data[glparam.propagation_param.refTP];
  TPDataType &tpdata_crnt = pData.tp_data[tp_crnt];

  // Set greedy parameters
  GreedyParameters param;
  GreedyApproach<3u, TReal> *GreedyAPI = new GreedyApproach<3u, TReal>();

  // Set images
  param.dim = 3;
  param.reslice_param.ref_image = "img_ref";

  // -- Set ref image
  auto casted_ref = CastToVectorImage<Image3DType, LinearIntensityMapping, TReal>(
        tpdata_crnt.img_srs);
  GreedyAPI->AddCachedInputObject(param.reslice_param.ref_image, casted_ref.GetPointer());

  // -- Set moving image
  std::string id_mov = "img_mov", id_res = "img_res";

  ResliceSpec rspec(id_mov, id_res, InterpSpec(InterpSpec::LABELWISE, 0.1));
  param.reslice_param.images.push_back(rspec);

  auto casted_mov = CastToVectorImage<LabelImageType, LinearIntensityMapping, TReal>(
        tpdata_ref.seg_srs);
  GreedyAPI->AddCachedInputObject(id_mov, casted_mov.GetPointer());

  // -- Set output image
  auto img_res = LabelImageType::New();
  tpdata_crnt.seg_srs = img_res.GetPointer();
  GreedyAPI->AddCachedOutputObject(id_res, img_res.GetPointer());


  std::map<std::string, itk::Object*> trans_spec_map;

  // Set the transformation chain
  for (size_t i = 0; i < tpdata_crnt.transform_specs.size(); ++i)
    {
    auto &trans_spec = tpdata_crnt.transform_specs[i];

    // -- Add affine
    // ---- Generate a unique id
    std::ostringstream oss;
    oss << "trans_" << i << "affine";
    std::string affine_id = oss.str();
    // ---- Save to map
    trans_spec_map[affine_id] = trans_spec.affine;
    // ---- Push to parameter
    param.reslice_param.transforms.push_back(TransformSpec(affine_id, -1.0));
    // ---- Add object to API cache
    GreedyAPI->AddCachedInputObject(affine_id, trans_spec.affine.GetPointer());

    // -- Add deformable
    // ---- Generate a unique id
    oss.clear();
    oss << "trans_" << i << "deform";
    std::string deform_id = oss.str();
    // ---- Save to map
    trans_spec_map[deform_id] = trans_spec.deform;
    // ---- Push to parameter
    param.reslice_param.transforms.push_back(TransformSpec(deform_id));
    // ---- Add object to API cache
    GreedyAPI->AddCachedInputObject(deform_id, trans_spec.deform.GetPointer());
    }

  int ret = GreedyAPI->RunReslice(param);

  std::cout << "-- Reslicing Completed: ret=" << ret << std::endl;
  delete GreedyAPI;
}

template <unsigned int VDim, typename TReal>
typename GreedyApproach<VDim, TReal>::Image3DPointer
GreedyApproach<VDim, TReal>
::ExtractTimePointImage(Image4DType *img4d, unsigned int tp)
{
  // Logic adapated from SNAP ImageWrapper method:
  // ConfigureTimePointImageFromImage4D()
  // Always use 1-based index for time point
  assert(tp > 0);
  printf("-- Extracting tp=%d \n", tp);

  unsigned int nt = img4d->GetBufferedRegion().GetSize()[3u];
  unsigned int bytes_per_volume = img4d->GetPixelContainer()->Size() / nt;

  Image3DPointer img3d = Image3DType::New();

  typename Image3DType::RegionType region;
  typename Image3DType::SpacingType spacing;
  typename Image3DType::PointType origin;
  typename Image3DType::DirectionType dir;
  for(unsigned int j = 0; j < 3; j++)
    {
    region.SetSize(j, img4d->GetBufferedRegion().GetSize()[j]);
    region.SetIndex(j, img4d->GetBufferedRegion().GetIndex()[j]);
    spacing[j] = img4d->GetSpacing()[j];
    origin[j] = img4d->GetOrigin()[j];
    for(unsigned int k = 0; k < 3; k++)
      dir(j,k) = img4d->GetDirection()(j,k);
    }

  // All of the information from the 4D image is propagaged to the 3D timepoints
  img3d->SetRegions(region);
  img3d->SetSpacing(spacing);
  img3d->SetOrigin(origin);
  img3d->SetDirection(dir);
  img3d->SetNumberOfComponentsPerPixel(img4d->GetNumberOfComponentsPerPixel());
  img3d->Allocate();

  // Set the buffer pointer
  img3d->GetPixelContainer()->SetImportPointer(
        img4d->GetBufferPointer() + bytes_per_volume * (tp - 1),
        bytes_per_volume);

  //img3d->Print(std::cout);

  return img3d;
}



template <unsigned int VDim, typename TReal>
void GreedyApproach<VDim, TReal>
::AddCachedInputObject(std::string key, itk::Object *object)
{
  m_ImageCache[key].target = object;
  m_ImageCache[key].force_write = false;
}

template <unsigned int VDim, typename TReal>
void GreedyApproach<VDim, TReal>
::AddCachedOutputObject(std::string key, itk::Object *object, bool force_write)
{
  m_ImageCache[key].target = object;
  m_ImageCache[key].force_write = force_write;
}

template <unsigned int VDim, typename TReal>
const typename GreedyApproach<VDim,TReal>::MetricLogType &
GreedyApproach<VDim,TReal>
::GetMetricLog() const
{
  return m_MetricLog;
}

template<unsigned int VDim, typename TReal>
MultiComponentMetricReport GreedyApproach<VDim, TReal>
::GetLastMetricReport() const
{
  // Find last non-empty result
  for(int k = m_MetricLog.size()-1; k >= 0; --k)
    {
    if(m_MetricLog[k].size())
      return m_MetricLog[k].back();
    }

  // If empty, throw exception
  throw GreedyException("Metric log is empty in GetLastMetricValue()");
  return MultiComponentMetricReport();
}

template <unsigned int VDim, typename TReal>
void GreedyApproach<VDim, TReal>
::ConfigThreads(const GreedyParameters &param)
{
  GreedyStdOut gout(param.verbosity);

  if(param.threads > 0)
    {
    gout.printf("Limiting the number of threads to %d\n", param.threads);
    itk::MultiThreaderBase::SetGlobalMaximumNumberOfThreads(param.threads);
    itk::MultiThreaderBase::SetGlobalDefaultNumberOfThreads(param.threads);
    }
  else
    {
    gout.printf("Executing with the default number of threads: %d\n",
                itk::MultiThreaderBase::GetGlobalDefaultNumberOfThreads());
    }
}

template<unsigned int VDim, typename TReal>
int GreedyApproach<VDim, TReal>
::Run(GreedyParameters &param)
{
  ConfigThreads(param);

  switch(param.mode)
    {
    case GreedyParameters::GREEDY:
      return Self::RunDeformable(param);
    case GreedyParameters::AFFINE:
      return Self::RunAffine(param);
    case GreedyParameters::BRUTE:
      return Self::RunBrute(param);
    case GreedyParameters::MOMENTS:
      return Self::RunAlignMoments(param);
    case GreedyParameters::RESLICE:
      return Self::RunReslice(param);
    case GreedyParameters::INVERT_WARP:
      return Self::RunInvertWarp(param);
    case GreedyParameters::JACOBIAN_WARP:
      return Self::RunJacobian(param);
    case GreedyParameters::ROOT_WARP:
      return Self::RunRootWarp(param);
    case GreedyParameters::METRIC:
      return Self::RunMetric(param);
    case GreedyParameters::PROPAGATION:
      return Self::RunPropagation(param);
    }

  return -1;
}




template class GreedyApproach<2, float>;
template class GreedyApproach<3, float>;
template class GreedyApproach<4, float>;
template class GreedyApproach<2, double>;
template class GreedyApproach<3, double>;
template class GreedyApproach<4, double>;
