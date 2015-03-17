/*=========================================================================

  Program:   ALFABIS fast image registration
  Language:  C++

  Copyright (c) Paul Yushkevich. All rights reserved.

  This program is part of ALFABIS

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
#ifndef MULTICOMPONENTNCCIMAGEMETRIC_H
#define MULTICOMPONENTNCCIMAGEMETRIC_H

#include "MultiComponentImageMetricBase.h"


/**
 * Normalized cross-correlation metric. This filter sets up a mini-pipeline with
 * a pre-compute filter that interpolates the moving image, N one-dimensional
 * mean filters, and a post-compute filter that generates the metric and the
 * gradient.
 */
template <class TMetricTraits>
class ITK_EXPORT MultiComponentNCCImageMetric :
    public MultiComponentImageMetricBase<TMetricTraits>
{
public:
  /** Standard class typedefs. */
  typedef MultiComponentNCCImageMetric<TMetricTraits>       Self;
  typedef MultiComponentImageMetricBase<TMetricTraits>      Superclass;
  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self)

  /** Run-time type information (and related methods) */
  itkTypeMacro( MultiComponentNCCImageMetric, MultiComponentImageMetricBase )

  /** Typedef to describe the output image region type. */
  typedef typename Superclass::OutputImageRegionType         OutputImageRegionType;

  /** Inherit some types from the superclass. */
  typedef typename Superclass::InputImageType                InputImageType;
  typedef typename Superclass::InputPixelType                InputPixelType;
  typedef typename Superclass::InputComponentType            InputComponentType;
  typedef typename Superclass::MetricImageType               MetricImageType;
  typedef typename Superclass::GradientImageType             GradientImageType;
  typedef typename Superclass::MaskImageType                 MaskImageType;


  typedef typename Superclass::IndexType                     IndexType;
  typedef typename Superclass::IndexValueType                IndexValueType;
  typedef typename Superclass::SizeType                      SizeType;
  typedef typename Superclass::SpacingType                   SpacingType;
  typedef typename Superclass::DirectionType                 DirectionType;
  typedef typename Superclass::ImageBaseType                 ImageBaseType;

  /** Information from the deformation field class */
  typedef typename Superclass::DeformationFieldType          DeformationFieldType;

  /** Determine the image dimension. */
  itkStaticConstMacro(ImageDimension, unsigned int, InputImageType::ImageDimension );

  /** Set the radius of the cross-correlation */
  itkSetMacro(Radius, SizeType)

  /** Get the radius of the cross-correlation */
  itkGetMacro(Radius, SizeType)

  /**
   * Set the working memory image for this filter. This function should be used to prevent
   * repeated allocation of memory when the metric is created/destructed in a loop. The
   * user can just pass in a pointer to a blank image, the filter will take care of allocating
   * the image as necessary
   */
  itkSetObjectMacro(WorkingImage, InputImageType)

  /** Summary results after running the filter */
  itkGetConstMacro(MetricValue, double)

  /**
   * Get the gradient scaling factor. To get the actual gradient of the metric, multiply the
   * gradient output of this filter by the scaling factor. Explanation: for efficiency, the
   * metrics return an arbitrarily scaled vector, such that adding the gradient to the
   * deformation field would INCREASE SIMILARITY. For metrics that are meant to be minimized,
   * this is the opposite of the gradient direction. For metrics that are meant to be maximized,
   * it is the gradient direction.
   */
  virtual double GetGradientScalingFactor() const { return 1.0; }


protected:
  MultiComponentNCCImageMetric() {}
  ~MultiComponentNCCImageMetric() {}

  /** SimpleWarpImageFilter is implemented as a multi-threaded filter.
   * As such, it needs to provide and implementation for
   * ThreadedGenerateData(). */
  void GenerateData();

private:
  MultiComponentNCCImageMetric(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  // A pointer to the working image. The user should supply this image in order to prevent
  // unnecessary memory allocation
  typename InputImageType::Pointer m_WorkingImage;

  // Radius of the cross-correlation
  SizeType m_Radius;

  // Vector of accumulated data (difference, gradient of affine transform, etc)
  double                          m_MetricValue;
};

#ifndef ITK_MANUAL_INSTANTIATION
#include "MultiComponentNCCImageMetric.txx"
#endif


#endif // MULTICOMPONENTNCCIMAGEMETRIC_H
