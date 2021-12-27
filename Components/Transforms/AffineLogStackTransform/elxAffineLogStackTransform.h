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
#include "itkStackTransform.h"
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
  /** Standard ITK-stuff. */
  typedef AffineLogStackTransform Self;
  typedef itk::AdvancedCombinationTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                            elx::TransformBase<TElastix>::FixedImageDimension>
                                        Superclass1;
  typedef elx::TransformBase<TElastix>  Superclass2;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

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

  typedef itk::AffineLogTransform<typename elx::TransformBase<TElastix>::CoordRepType, Self::SpaceDimension>
                                                          AffineLogTransformType;
  typedef typename AffineLogTransformType::Pointer        AffineLogTransformPointer;
  typedef typename AffineLogTransformType::InputPointType InputPointType;

  /** The ITK-class for the sub transforms, which have a reduced dimension. */
  typedef itk::AffineLogTransform<typename elx::TransformBase<TElastix>::CoordRepType, Self::ReducedSpaceDimension>
                                                                       ReducedDimensionAffineLogTransformBaseType;
  typedef typename ReducedDimensionAffineLogTransformBaseType::Pointer ReducedDimensionAffineLogTransformBasePointer;

  typedef typename ReducedDimensionAffineLogTransformBaseType::OutputVectorType ReducedDimensionOutputVectorType;
  typedef typename ReducedDimensionAffineLogTransformBaseType::InputPointType   ReducedDimensionInputPointType;

  /** Typedefs inherited from the superclass. */
  using typename Superclass1::ParametersType;
  using typename Superclass1::NumberOfParametersType;

  /** Typedef's from TransformBase. */
  using typename Superclass2::ElastixType;
  using typename Superclass2::ElastixPointer;
  using typename Superclass2::ParameterMapType;
  using typename Superclass2::ConfigurationType;
  using typename Superclass2::ConfigurationPointer;
  using typename Superclass2::RegistrationType;
  using typename Superclass2::RegistrationPointer;
  using typename Superclass2::CoordRepType;
  using typename Superclass2::FixedImageType;
  using typename Superclass2::MovingImageType;
  typedef typename Superclass2::ITKBaseType              ITKBaseType;
  typedef typename Superclass2::CombinationTransformType CombinationTransformType;

  /** Reduced Dimension typedef's. */
  typedef float                                              PixelType;
  typedef itk::Image<PixelType, Self::ReducedSpaceDimension> ReducedDimensionImageType;
  typedef itk::ImageRegion<Self::ReducedSpaceDimension>      ReducedDimensionRegionType;
  typedef typename ReducedDimensionImageType::PointType      ReducedDimensionPointType;
  typedef typename ReducedDimensionImageType::SizeType       ReducedDimensionSizeType;
  typedef typename ReducedDimensionRegionType::IndexType     ReducedDimensionIndexType;
  typedef typename ReducedDimensionImageType::SpacingType    ReducedDimensionSpacingType;
  typedef typename ReducedDimensionImageType::DirectionType  ReducedDimensionDirectionType;
  typedef typename ReducedDimensionImageType::PointType      ReducedDimensionOriginType;

  /** For scales setting in the optimizer */
  using typename Superclass2::ScalesType;

  /** Other typedef's. */
  typedef typename FixedImageType::IndexType                                 IndexType;
  typedef typename FixedImageType::SizeType                                  SizeType;
  typedef typename FixedImageType::PointType                                 PointType;
  typedef typename FixedImageType::SpacingType                               SpacingType;
  typedef typename FixedImageType::RegionType                                RegionType;
  typedef typename FixedImageType::DirectionType                             DirectionType;
  typedef typename itk::ContinuousIndex<CoordRepType, ReducedSpaceDimension> ReducedDimensionContinuousIndexType;
  typedef typename itk::ContinuousIndex<CoordRepType, SpaceDimension>        ContinuousIndexType;

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
  AffineLogStackTransform() = default;

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
  AffineLogStackTransform(const Self &) = delete;
  void
  operator=(const Self &) = delete;

  /** Typedef for stack transform. */
  typedef itk::StackTransform<ElastixBase::CoordRepType, SpaceDimension, SpaceDimension> StackTransformType;

  /** The Affine stack transform. */
  typename StackTransformType::Pointer m_StackTransform;

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
