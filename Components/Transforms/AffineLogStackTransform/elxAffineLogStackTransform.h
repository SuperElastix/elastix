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

#ifndef ELXAFFINELOGSTACKTRANSFORM_H
#define ELXAFFINELOGSTACKTRANSFORM_H

/** Include itk transforms needed. */
#include "itkAdvancedCombinationTransform.h"
#include "itkAffineLogStackTransform.h"
#include "../AffineLogTransform/itkAffineLogTransform.h"

#include "elxIncludes.h"

namespace elastix
{

/**
 * \class AffineLogStackTransform
 * \brief An affine log transform based on the itkStackTransform.
 *
 *
 * \ingroup Transforms
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT AffineLogStackTransform
  : public itk::AdvancedCombinationTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                             elx::TransformBase<TElastix>::FixedImageDimension>
  , public elx::TransformBase<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(AffineLogStackTransform);

  /** Standard ITK-stuff. */
  using Self = AffineLogStackTransform;
  using Superclass1 = itk::AdvancedCombinationTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                                        elx::TransformBase<TElastix>::FixedImageDimension>;
  using Superclass2 = elx::TransformBase<TElastix>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(AffineLogStackTransform, itk::AdvancedCombinationTransform);

  /** Name of this class.
   * Use this name in the parameter file to select this specific transform. \n
   * example: <tt>(Transform "AffineStackTransform")</tt>\n
   */
  elxClassNameMacro("AffineLogStackTransform");

  /** (Reduced) dimension of the fixed image. */
  itkStaticConstMacro(SpaceDimension, unsigned int, Superclass2::FixedImageDimension);
  itkStaticConstMacro(ReducedSpaceDimension, unsigned int, Superclass2::FixedImageDimension - 1);

  using AffineLogTransformType =
    itk::AffineLogTransform<typename elx::TransformBase<TElastix>::CoordRepType, Self::SpaceDimension>;
  using AffineLogTransformPointer = typename AffineLogTransformType::Pointer;
  using InputPointType = typename AffineLogTransformType::InputPointType;

  /** The ITK-class for the sub transforms, which have a reduced dimension. */
  using ReducedDimensionAffineLogTransformBaseType =
    itk::AffineLogTransform<typename elx::TransformBase<TElastix>::CoordRepType, Self::ReducedSpaceDimension>;
  using ReducedDimensionAffineLogTransformBasePointer = typename ReducedDimensionAffineLogTransformBaseType::Pointer;

  using ReducedDimensionOutputVectorType = typename ReducedDimensionAffineLogTransformBaseType::OutputVectorType;
  using ReducedDimensionInputPointType = typename ReducedDimensionAffineLogTransformBaseType::InputPointType;

  /** Typedefs inherited from the superclass. */
  using typename Superclass1::ParametersType;
  using typename Superclass1::NumberOfParametersType;

  /** Typedef's from TransformBase. */
  using typename Superclass2::ElastixType;
  using typename Superclass2::ParameterMapType;
  using typename Superclass2::RegistrationType;
  using typename Superclass2::CoordRepType;
  using typename Superclass2::FixedImageType;
  using typename Superclass2::MovingImageType;
  using ITKBaseType = typename Superclass2::ITKBaseType;
  using CombinationTransformType = typename Superclass2::CombinationTransformType;

  /** Reduced Dimension typedef's. */
  using PixelType = float;
  using ReducedDimensionImageType = itk::Image<PixelType, Self::ReducedSpaceDimension>;
  using ReducedDimensionRegionType = itk::ImageRegion<Self::ReducedSpaceDimension>;
  using ReducedDimensionPointType = typename ReducedDimensionImageType::PointType;
  using ReducedDimensionSizeType = typename ReducedDimensionImageType::SizeType;
  using ReducedDimensionIndexType = typename ReducedDimensionRegionType::IndexType;
  using ReducedDimensionSpacingType = typename ReducedDimensionImageType::SpacingType;
  using ReducedDimensionDirectionType = typename ReducedDimensionImageType::DirectionType;
  using ReducedDimensionOriginType = typename ReducedDimensionImageType::PointType;

  /** For scales setting in the optimizer */
  using typename Superclass2::ScalesType;

  /** Other typedef's. */
  using IndexType = typename FixedImageType::IndexType;
  using SizeType = typename FixedImageType::SizeType;
  using PointType = typename FixedImageType::PointType;
  using SpacingType = typename FixedImageType::SpacingType;
  using RegionType = typename FixedImageType::RegionType;
  using DirectionType = typename FixedImageType::DirectionType;
  using ReducedDimensionContinuousIndexType = typename itk::ContinuousIndex<CoordRepType, ReducedSpaceDimension>;
  using ContinuousIndexType = typename itk::ContinuousIndex<CoordRepType, SpaceDimension>;

  /** Execute stuff before anything else is done:*/

  int
  BeforeAll() override;

  /** Execute stuff before the actual registration:
   * \li Set the stack transform parameters.
   * \li Set initial sub transforms.
   * \li Create initial registration parameters.
   */
  void
  BeforeRegistration() override;

  /** Set the scales
   * \li If AutomaticScalesEstimation is "true" estimate scales
   * \li If scales are provided by the user use those,
   * \li Otherwise use some default value
   * This function is called by BeforeRegistration, after
   * the InitializeTransform function is called
   */
  virtual void
  SetScales();

  /** Function to read transform-parameters from a file. */
  void
  ReadFromFile() override;

protected:
  /** The constructor. */
  AffineLogStackTransform() { this->Superclass1::SetCurrentTransform(m_StackTransform); }

  /** The destructor. */
  ~AffineLogStackTransform() override = default;

  /** Try to read the CenterOfRotationPoint from the transform parameter file
   * The CenterOfRotationPoint is already in world coordinates.
   * Transform parameter files generated by elastix version > 3.402
   * save the center of rotation in this way.
   */
  virtual bool
  ReadCenterOfRotationPoint(ReducedDimensionInputPointType & rotationPoint) const;

private:
  elxOverrideGetSelfMacro;

  /** Method initialize the parameters (to 0). */
  void
  InitializeTransform();

  /** Creates a map of the parameters specific for this (derived) transform type. */
  ParameterMapType
  CreateDerivedTransformParametersMap() const override;

  /** The deleted copy constructor and assignment operator. */
  /** Typedef for stack transform. */
  using StackTransformType = itk::AffineLogStackTransform<SpaceDimension>;

  /** The Affine stack transform. */
  const typename StackTransformType::Pointer m_StackTransform{ StackTransformType::New() };

  /** Dummy sub transform to be used to set sub transforms of stack transform. */
  ReducedDimensionAffineLogTransformBasePointer m_DummySubTransform;

  /** Stack variables. */
  unsigned int m_NumberOfSubTransforms;
  double       m_StackOrigin, m_StackSpacing;

  /** Initialize the affine transform. */
  unsigned int
  InitializeAffineLogTransform();
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxAffineLogStackTransform.hxx"
#endif

#endif // ELXAFFINELOGSTACKTRANSFORM_H
