/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
/*=========================================================================

Program:   Insight Segmentation & Registration Toolkit
Module:    $RCSfile: itkAdvancedRayCastInterpolateImageFunction.h,v $
Language:  C++
Date:      $Date: 2009-04-23 03:53:36 $
Version:   $Revision: 1.17 $

Copyright (c) Insight Software Consortium. All rights reserved.
See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef itkAdvancedRayCastInterpolateImageFunction_h
#define itkAdvancedRayCastInterpolateImageFunction_h

#include "itkInterpolateImageFunction.h"
#include "itkTransform.h"
#include "itkVector.h"

namespace itk
{

/** \class AdvancedRayCastInterpolateImageFunction
 * \brief Projective interpolation of an image at specified positions.
 *
 * AdvancedRayCastInterpolateImageFunction casts rays through a 3-dimensional
 * image and uses bilinear interpolation to integrate each plane of
 * voxels traversed.
 *
 * \warning This interpolator works for 3-dimensional images only.
 *
 * \ingroup ImageFunctions
 */
template <class TInputImage, class TCoordRep = double>
class ITK_TEMPLATE_EXPORT AdvancedRayCastInterpolateImageFunction
  : public InterpolateImageFunction<TInputImage, TCoordRep>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(AdvancedRayCastInterpolateImageFunction);

  /** Standard class typedefs. */
  using Self = AdvancedRayCastInterpolateImageFunction;
  using Superclass = InterpolateImageFunction<TInputImage, TCoordRep>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Constants for the image dimensions */
  itkStaticConstMacro(InputImageDimension, unsigned int, TInputImage::ImageDimension);

  /**
   * Type of the Transform Base class
   * The fixed image should be a 3D image
   */
  using TransformType = Transform<TCoordRep, InputImageDimension, InputImageDimension>;

  using TransformPointer = typename TransformType::Pointer;
  using InputPointType = typename TransformType::InputPointType;
  using OutputPointType = typename TransformType::OutputPointType;
  using TransformParametersType = typename TransformType::ParametersType;
  using TransformJacobianType = typename TransformType::JacobianType;

  using PixelType = typename Superclass::InputPixelType;

  using SizeType = typename TInputImage::SizeType;

  using DirectionType = Vector<TCoordRep, InputImageDimension>;

  /**  Type of the Interpolator Base class */
  using InterpolatorType = InterpolateImageFunction<TInputImage, TCoordRep>;

  using InterpolatorPointer = typename InterpolatorType::Pointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro(AdvancedRayCastInterpolateImageFunction, InterpolateImageFunction);

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** OutputType typedef support. */
  using typename Superclass::OutputType;

  /** InputImageType typedef support. */
  using typename Superclass::InputImageType;

  /** RealType typedef support. */
  using typename Superclass::RealType;

  /** Dimension underlying input image. */
  itkStaticConstMacro(ImageDimension, unsigned int, Superclass::ImageDimension);

  /** Point typedef support. */
  using typename Superclass::PointType;

  /** Index typedef support. */
  using typename Superclass::IndexType;

  /** ContinuousIndex typedef support. */
  using typename Superclass::ContinuousIndexType;

  /** \brief
   * Interpolate the image at a point position.
   *
   * Returns the interpolated image intensity at a
   * specified point position. No bounds checking is done.
   * The point is assume to lie within the image buffer.
   *
   * ImageFunction::IsInsideBuffer() can be used to check bounds before
   * calling the method.
   */
  OutputType
  Evaluate(const PointType & point) const override;

  /** Interpolate the image at a continuous index position
   *
   * Returns the interpolated image intensity at a
   * specified index position. No bounds checking is done.
   * The point is assume to lie within the image buffer.
   *
   * Subclasses must override this method.
   *
   * ImageFunction::IsInsideBuffer() can be used to check bounds before
   * calling the method.
   */
  OutputType
  EvaluateAtContinuousIndex(const ContinuousIndexType & index) const override;

  /** Connect the Transform. */
  itkSetObjectMacro(Transform, TransformType);
  /** Get a pointer to the Transform.  */
  itkGetModifiableObjectMacro(Transform, TransformType);

  /** Connect the Interpolator. */
  itkSetObjectMacro(Interpolator, InterpolatorType);
  /** Get a pointer to the Interpolator.  */
  itkGetModifiableObjectMacro(Interpolator, InterpolatorType);

  /** Connect the Interpolator. */
  itkSetMacro(FocalPoint, InputPointType);
  /** Get a pointer to the Interpolator.  */
  itkGetConstMacro(FocalPoint, InputPointType);

  /** Connect the Transform. */
  itkSetMacro(Threshold, double);
  /** Get a pointer to the Transform.  */
  itkGetConstMacro(Threshold, double);

  /** Check if a point is inside the image buffer.
   * \warning For efficiency, no validity checking of
   * the input image pointer is done. */
  inline bool
  IsInsideBuffer(const PointType &) const override
  {
    return true;
  }


  bool
  IsInsideBuffer(const ContinuousIndexType &) const override
  {
    return true;
  }


  bool
  IsInsideBuffer(const IndexType &) const override
  {
    return true;
  }


protected:
  /// Constructor
  AdvancedRayCastInterpolateImageFunction() = default;

  /// Destructor
  ~AdvancedRayCastInterpolateImageFunction() override = default;

  /// Print the object
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /// Transformation used to calculate the new focal point position
  TransformPointer m_Transform;

  /// The focal point or position of the ray source
  InputPointType m_FocalPoint{};

  /// The threshold above which voxels along the ray path are integrated.
  double m_Threshold{ 0.0 };

  /// Pointer to the interpolator
  InterpolatorPointer m_Interpolator;

private:
  SizeType
  GetRadius() const override
  {
    const InputImageType * const input = this->GetInputImage();
    if (!input)
    {
      itkExceptionMacro("Input image required!");
    }
    return input->GetLargestPossibleRegion().GetSize();
  }
};

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkAdvancedRayCastInterpolateImageFunction.hxx"
#endif

#endif
