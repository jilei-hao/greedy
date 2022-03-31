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
#ifndef GREEDYAPI_H
#define GREEDYAPI_H

#include "GreedyParameters.h"
#include "GreedyException.h"
#include "MultiComponentMetricReport.h"
#include "lddmm_data.h"
#include "AffineCostFunctions.h"
#include <vnl/vnl_random.h>
#include <map>
#include "itkCommand.h"
#include <vtkSmartPointer.h>
#include "itkMatrixOffsetTransformBase.h"
#include "itkImageRegionIterator.h"

template <typename T, unsigned int V> class MultiImageOpticalFlowHelper;

namespace itk {
  template <typename T, unsigned int D1, unsigned int D2> class MatrixOffsetTransformBase;

}

class vtkPolyData;

template <typename TReal>
struct PropagationSegGroup
{
  typedef itk::Image<TReal, 4u> Image4DType;
  typedef typename Image4DType::Pointer Image4DPointer;
  typedef itk::Image<TReal, 3u> Image3DType;
  typedef typename Image3DType::Pointer Image3DPointer;
  typedef itk::Image<short, 3u> LabelImageType;
  typedef typename LabelImageType::Pointer LabelImagePointer;
  typedef vtkSmartPointer<vtkPolyData> MeshPointer;

  LabelImagePointer seg_ref;
  LabelImagePointer seg_ref_srs;
  MeshPointer mesh_ref;
  std::string outdir;
};

struct PropagationMeshGroup
{

};

template <typename TReal>
struct TimePointTransformSpec
{
  typedef itk::MatrixOffsetTransformBase<double, 3u, 3u> TransformType;
  typedef LDDMMData<TReal, 3u> LDDMM3DType;
  typedef typename LDDMM3DType::VectorImageType VectorImage3DType;
  typedef typename VectorImage3DType::Pointer VectorImage3DPointer;

  TimePointTransformSpec(TransformType::Pointer _affine, VectorImage3DPointer _deform)
    : affine(_affine), deform (_deform) {}

  TimePointTransformSpec(TransformType::Pointer _affine)
    : affine(_affine), deform (nullptr) {}

  TransformType::Pointer affine;
  VectorImage3DPointer deform;
};

template <typename TReal>
struct TimePointData
{
  typedef itk::Image<TReal, 4u> Image4DType;
  typedef typename Image4DType::Pointer Image4DPointer;
  typedef itk::Image<TReal, 3u> Image3DType;
  typedef typename Image3DType::Pointer Image3DPointer;
  typedef LDDMMData<TReal, 3u> LDDMM3DType;
  typedef typename LDDMM3DType::VectorImageType VectorImage3DType;
  typedef typename VectorImage3DType::Pointer VectorImage3DPointer;
  typedef itk::Image<short, 3u> LabelImageType;
  typedef typename LabelImageType::Pointer LabelImagePointer;
  typedef itk::MatrixOffsetTransformBase<double, 3u, 3u> TransformType;
  typedef vtkSmartPointer<vtkPolyData> MeshPointer;

  TimePointData();

  // This method is to convert label image to double image
  // so the the image can be read as fixed mask
  static Image3DPointer CastLabelToDoubleImage(LabelImageType *input)
  {
    auto output = Image3DType::New();
    output->SetRegions(input->GetLargestPossibleRegion());
    output->SetDirection(input->GetDirection());
    output->SetOrigin(input->GetOrigin());
    output->SetSpacing(input->GetSpacing());
    output->Allocate();

    itk::ImageRegionIterator<LabelImageType> it_input(
          input, input->GetLargestPossibleRegion());
    itk::ImageRegionIterator<Image3DType> it_output(
          output, output->GetLargestPossibleRegion());

    // Deep copy pixels
    while (!it_input.IsAtEnd())
      {
      it_output.Set(it_input.Get());
      ++it_output;
      ++it_input;
      }

    return output;
  }

  static LabelImagePointer ResliceLabelImageWithIdentityMatrix(
      Image3DType *ref, LabelImageType *src);

  static MeshPointer GetMeshFromLabelImage(LabelImageType *img);

  static LabelImagePointer TrimLabelImage(LabelImageType *input, double vox);

  Image3DPointer img;
  Image3DPointer img_srs;
  LabelImagePointer seg;
  LabelImagePointer seg_srs;
  MeshPointer mesh;
  MeshPointer mesh_srs;
  LabelImagePointer full_res_mask;
  TransformType::Pointer affine_to_prev;
  VectorImage3DPointer deform_to_prev;
  VectorImage3DPointer deform_to_ref;
  VectorImage3DPointer deform_from_prev;
  VectorImage3DPointer deform_from_ref;
  std::vector<TimePointTransformSpec<TReal>> transform_specs;
  std::vector<TimePointTransformSpec<TReal>> full_res_label_trans_specs;
};

template <typename TReal>
struct PropagationData
{
  typedef itk::Image<TReal, 4u> Image4DType;
  typedef typename Image4DType::Pointer Image4DPointer;
  typedef itk::Image<TReal, 3u> Image3DType;
  typedef typename Image3DType::Pointer Image3DPointer;

  Image4DPointer img4d;
  // Only store data for current output list
  std::map<unsigned int, TimePointData<TReal>> tp_data;
  std::vector<PropagationSegGroup<TReal>> seg_list;
  std::vector<PropagationMeshGroup> mesh_list;
  Image3DPointer full_res_ref_space;

};

/**
 * This is the top level class for the greedy software. It contains methods
 * for deformable and affine registration.
 */
template <unsigned int VDim, typename TReal = double>
class GreedyApproach
{
public:

  typedef GreedyApproach<VDim, TReal> Self;

  typedef LDDMMData<TReal, VDim> LDDMMType;
  typedef typename LDDMMType::ImageBaseType ImageBaseType;
  typedef typename LDDMMType::ImageType ImageType;
  typedef typename LDDMMType::ImagePointer ImagePointer;
  typedef typename LDDMMType::VectorImageType VectorImageType;
  typedef typename LDDMMType::VectorImagePointer VectorImagePointer;
  typedef typename LDDMMType::CompositeImageType CompositeImageType;
  typedef typename LDDMMType::CompositeImagePointer CompositeImagePointer;

  // Typedefs for propagation
  typedef itk::Image<TReal, 4u> Image4DType;
  typedef typename Image4DType::Pointer Image4DPointer;
  typedef itk::Image<TReal, 3u> Image3DType;
  typedef typename Image3DType::Pointer Image3DPointer;
  typedef itk::Image<short, 3u> LabelImageType;
  typedef typename LabelImageType::Pointer LabelImagePointer;

  typedef vnl_vector_fixed<TReal, VDim> VecFx;
  typedef vnl_matrix_fixed<TReal, VDim, VDim> MatFx;

  typedef std::vector< std::vector<MultiComponentMetricReport> > MetricLogType;

  typedef MultiImageOpticalFlowHelper<TReal, VDim> OFHelperType;

  typedef itk::MatrixOffsetTransformBase<TReal, VDim, VDim> LinearTransformType;

  struct ImagePair {
    ImagePointer fixed, moving;
    VectorImagePointer grad_moving;
    double weight;
  };

  // Mesh data structures
  typedef vtkSmartPointer<vtkPolyData> MeshPointer;
  typedef std::vector<MeshPointer> MeshArray;

  static void ConfigThreads(const GreedyParameters &param);

  int Run(GreedyParameters &param);

  int RunDeformable(GreedyParameters &param);

  int RunAffine(GreedyParameters &param);

  int RunBrute(GreedyParameters &param);

  int RunReslice(GreedyParameters &param);

  int RunInvertWarp(GreedyParameters &param);

  int RunRootWarp(GreedyParameters &param);

  int RunAlignMoments(GreedyParameters &param);

  int RunJacobian(GreedyParameters &param);

  int RunMetric(GreedyParameters &param);

  int RunPropagation(GreedyParameters &param);

  int ComputeMetric(GreedyParameters &param, MultiComponentMetricReport &metric_report);

  /**
   * Add an image that is already in memory to the internal cache, and
   * associate it with a filename. This provides a way for images already
   * loaded in memory to be passed in to the Greedy API while using the
   * standard parameter structures.
   *
   * Normally, images such as the fixed image are passed as part of the
   * GreedyParameters object as filenames. For example, we might set
   *
   *   param.inputs[0].fixed = "/tmp/goo.nii.gz";
   *
   * However, if we are linking to the greedy API from another program and
   * already have the fixed image in memory, we can use the cache mechanism
   * instead.
   *
   *   greedyapi.AddCachedInputObject("FIXED-0", myimage);
   *   param.inputs[0].fixed = "FIXED-0";
   *
   * The API will check the cache before loading the image. The type of the
   * object in the cache must match the type of the object expected internally,
   * which is VectorImage for most images. If not, an exception will be
   * thrown.
   *
   * Note that the cache does not use smart pointers to refer to the objects
   * so it's the caller's responsibility to keep the object pointed to while
   * the API is being used.
   */
  void AddCachedInputObject(std::string key, itk::Object *object);

  void AddCachedInputObject(std::string key, vtkPolyData *object);

  /**
   * Add an image/matrix to the output cache. This has the same behavior as
   * the input cache, but there is an additional flag as to whether you want
   * to save the output object to the specified filename in addition to writing
   * it to the cached image/matrix. This allows you to both store the result in
   * the cache and write it to a filename specified in the key
   */
  void AddCachedOutputObject(std::string key, itk::Object *object, bool force_write = false);

  void AddCachedOutputObject(std::string key, vtkPolyData *object, bool force_write = false);

  /**
   * Get the metric log - values of metric per level. Can be called from
   * callback functions and observers
   */
  const MetricLogType &GetMetricLog() const;

  /** Get the last value of the metric recorded */
  MultiComponentMetricReport GetLastMetricReport() const;

  vnl_matrix<double> ReadAffineMatrixViaCache(const TransformSpec &ts);

  void WriteAffineMatrixViaCache(const std::string &filename, const vnl_matrix<double> &Qp);

  static vnl_matrix<double> ReadAffineMatrix(const TransformSpec &ts);

  /**
   * Helper method to read an affine matrix from file into an ITK transform type
   */
  static void ReadAffineTransform(const TransformSpec &ts, LinearTransformType *tran);

  static void WriteAffineMatrix(const std::string &filename, const vnl_matrix<double> &Qp);

  /**
   * Helper method to write an affine ITK transform type to a matrix file
   */
  static void WriteAffineTransform(const std::string &filename, LinearTransformType *tran);

  static vnl_matrix<double> MapAffineToPhysicalRASSpace(
      OFHelperType &of_helper, unsigned int group, unsigned int level,
      LinearTransformType *tran);

  static void MapPhysicalRASSpaceToAffine(
      OFHelperType &of_helper, unsigned int group, unsigned int level,
      vnl_matrix<double> &Qp,
      LinearTransformType *tran);

  static void MapRASAffineToPhysicalWarp(const vnl_matrix<double> &mat,
                                         VectorImagePointer &out_warp);

  void RecordMetricValue(const MultiComponentMetricReport &metric);

  // Helper method to print iteration reports
  std::string PrintIter(int level, int iter, const MultiComponentMetricReport &metric) const;

  /**
   * Read images specified in parameters into a helper data structure and initialize
   * the multi-resolution pyramid
   */
  void ReadImages(GreedyParameters &param, OFHelperType &ofhelper,
                  bool force_resample_to_fixed_space);

  /**
   * Compute one of the metrics (specified in the parameters). This code is called by
   * RunDeformable and is provided as a separate public method for testing purposes
   */
  void EvaluateMetricForDeformableRegistration(
      GreedyParameters &param, OFHelperType &of_helper, unsigned int level,
      VectorImageType *phi, MultiComponentMetricReport &metric_report,
      ImageType *out_metric_image, VectorImageType *out_metric_gradient, double eps);

  /**
   * Load initial transform (affine or deformable) into a deformation field
   */
  void LoadInitialTransform(
      GreedyParameters &param, OFHelperType &of_helper,
      unsigned int level, VectorImageType *phi);


  /**
   * Generate an affine cost function for given level based on parameter values
   */
  AbstractAffineCostFunction<VDim, TReal> *CreateAffineCostFunction(
      GreedyParameters &param, OFHelperType &of_helper, int level);

  /**
   * Initialize affine transform (to identity, filename, etc.) based on the
   * parameter values; resulting transform is placed into tLevel.
   */
  void InitializeAffineTransform(
      GreedyParameters &param, OFHelperType &of_helper,
      AbstractAffineCostFunction<VDim, TReal> *acf,
      LinearTransformType *tLevel);

  /**
   * Check the derivatives of affine transform
   */
  int CheckAffineDerivatives(GreedyParameters &param, OFHelperType &of_helper,
                             AbstractAffineCostFunction<VDim, TReal> *acf,
                             LinearTransformType *tLevel, int level, double tol);

  /** Apply affine transformation to a mesh */
  static void TransformMeshAffine(vtkPolyData *mesh, vnl_matrix<double> mat);

  /** Apply warp to a mesh */
  static void TransformMeshWarp(vtkPolyData *mesh, VectorImageType *warp);

protected:

  struct CacheEntry {
    itk::Object *target;
    bool force_write;
  };

  typedef std::map<std::string, CacheEntry> ImageCache;
  ImageCache m_ImageCache;

  struct PolyDataCacheEntry {
    vtkSmartPointer<vtkPolyData> target;
    bool force_write;
  };

  typedef std::map<std::string, PolyDataCacheEntry> PolyDataCache;
  PolyDataCache m_PolyDataCache;

  // A log of metric values used during registration - so metric can be looked up
  // in the callbacks to RunAffine, etc.
  MetricLogType m_MetricLog;

  // This function reads the image from disk, or from a memory location mapped to a
  // string. The first approach is used by the command-line interface, and the second
  // approach is used by the API, allowing images to be passed from other software.
  // An optional second argument is used to store the component type, but only if
  // the image is actually loaded from disk. For cached images, the component type
  // will be unknown.
  template <class TImage>
  itk::SmartPointer<TImage> ReadImageViaCache(const std::string &filename,
                                              itk::IOComponentEnum *comp_type = NULL);

  template<class TObject> TObject *CheckCache(const std::string &filename) const;

  // Get a filename for dumping intermediate outputs
  std::string GetDumpFile(const GreedyParameters &param, const char *pattern, ...);

  // This function reads an image base object via cache. It is more permissive than using
  // ReadImageViaCache.
  typename ImageBaseType::Pointer ReadImageBaseViaCache(const std::string &filename);

  // These functions read/write vtkPolyData object via cache, or from disk
  vtkSmartPointer<vtkPolyData> ReadPolyDataViaCache(const std::string &filename);
  void WritePolyDataViaCache(vtkPolyData *mesh, const std::string &filename);

  // Write an image using the cache
  template <class TImage>
  void WriteImageViaCache(TImage *img, const std::string &filename,
                          itk::IOComponentEnum comp = itk::IOComponentEnum::UNKNOWNCOMPONENTTYPE);

  // Write a compressed warp via cache (in float format)
  void WriteCompressedWarpInPhysicalSpaceViaCache(
    ImageBaseType *moving_ref_space, VectorImageType *warp, const char *filename, double precision);

  void ReadTransformChain(const std::vector<TransformSpec> &tran_chain,
                          ImageBaseType *ref_space,
                          VectorImagePointer &out_warp,
                          MeshArray *meshes = nullptr);

  // Compute the moments of a composite image (mean and covariance matrix of coordinate weighted by intensity)
  void ComputeImageMoments(CompositeImageType *image, const vnl_vector<float> &weights, VecFx &m1, MatFx &m2);

  // Resample an image to reference space if the spaces do not match or if an explicit warp is provided
  CompositeImagePointer ResampleImageToReferenceSpaceIfNeeded(
      CompositeImageType *img, ImageBaseType *ref_space, VectorImageType *resample_warp, TReal fill_value);

  ImagePointer ResampleMaskToReferenceSpaceIfNeeded(
      ImageType *mask, ImageBaseType *ref_space, VectorImageType *resample_warp);

  // Extract 3D image from given time point of the 4D Image
  static Image3DPointer ExtractTimePointImage(Image4DType *img4d, unsigned int tp);

  // Propagation affine run
  static void RunPropagationAffine(GreedyParameters &glparam, PropagationData<TReal> &pData
                                   ,unsigned int tp_fix, unsigned int tp_mov);

  // Propagation deform run
  static void RunPropagationDeformable(GreedyParameters &glparam, PropagationData<TReal> &pData
                                   ,unsigned int tp_fix, unsigned int tp_mov, bool isFullRes = false);

  // Propagation reslice run
  static void RunPropagationReslice(GreedyParameters &glparam, PropagationData<TReal> &pData
                                   ,unsigned int tp_mov, unsigned int tp_ref, bool isFullRes = false);

  // Propagation reslice run
  static void RunPropagationMeshReslice(GreedyParameters &glparam, PropagationData<TReal> &pData
                                   ,unsigned int tp_mov, unsigned int tp_ref);

  // friend class PureAffineCostFunction<VDim, TReal>;

};


#endif // GREEDYAPI_H
