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
#ifndef _ELXAFFINELOGTRANSFORM_H_
#define _ELXAFFINELOGTRANSFORM_H_

#include "itkAdvancedCombinationTransform.h"
#include "itkAffineLogTransform.h"
#include "itkCenteredTransformInitializer.h"
#include "elxIncludes.h"

namespace elastix
{

/**
 * \class AffineLogTransformElastix
 * \brief
 *
 * This transform is an affine transformation, with a different parametrisation
 * than the usual one.
 *
 * \warning: the behaviour of this transform might still change in the future. It is still experimental.
 *
 * \ingroup Transforms
 * \sa AffineLogTransform
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT AffineLogTransformElastix
  : public itk::AdvancedCombinationTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                             elx::TransformBase<TElastix>::FixedImageDimension>
  , public elx::TransformBase<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(AffineLogTransformElastix);

  /** Standard ITK-stuff.*/
  using Self = AffineLogTransformElastix;
  using Superclass1 = itk::AdvancedCombinationTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                                        elx::TransformBase<TElastix>::FixedImageDimension>;
  using Superclass2 = elx::TransformBase<TElastix>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** The ITK-class that provides most of the functionality, and
   * that is set as the "CurrentTransform" in the CombinationTransform */
  using AffineLogTransformType = itk::AffineLogTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                                         elx::TransformBase<TElastix>::FixedImageDimension>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(AffineLogTransformElastix, AdvancedCombinationTransform);

  /** Name of this class.
   * Use this name in the parameter file to select this specific transform. \n
   * example: <tt>(Transform "AffineLogTransform")</tt>\n
   */
  elxClassNameMacro("AffineLogTransform");

  /** Dimension of the fixed image. */
  itkStaticConstMacro(SpaceDimension, unsigned int, Superclass2::FixedImageDimension);

  /** Typedefs inherited from the superclass. */
  using typename Superclass1::ScalarType;
  using typename Superclass1::ParametersType;
  using typename Superclass1::NumberOfParametersType;
  using typename Superclass1::JacobianType;

  using typename Superclass1::InputPointType;
  using typename Superclass1::OutputPointType;
  using typename Superclass1::InputVectorType;
  using typename Superclass1::OutputVectorType;
  using typename Superclass1::InputCovariantVectorType;
  using typename Superclass1::OutputCovariantVectorType;
  using typename Superclass1::InputVnlVectorType;
  using typename Superclass1::OutputVnlVectorType;

  using AffineLogTransformPointer = typename AffineLogTransformType::Pointer;
  using OffsetType = typename AffineLogTransformType::OffsetType;

  /** Typedef's inherited from TransformBase. */
  using typename Superclass2::ElastixType;
  using typename Superclass2::ParameterMapType;
  using typename Superclass2::RegistrationType;
  using typename Superclass2::CoordRepType;
  using typename Superclass2::FixedImageType;
  using typename Superclass2::MovingImageType;
  using ITKBaseType = typename Superclass2::ITKBaseType;
  using CombinationTransformType = typename Superclass2::CombinationTransformType;

  /** Other typedef's. */
  using IndexType = typename FixedImageType::IndexType;
  using IndexValueType = typename IndexType::IndexValueType;
  using SizeType = typename FixedImageType::SizeType;
  using PointType = typename FixedImageType::PointType;
  using SpacingType = typename FixedImageType::SpacingType;
  using RegionType = typename FixedImageType::RegionType;
  using DirectionType = typename FixedImageType::DirectionType;

  using TransformInitializerType =
    itk::CenteredTransformInitializer<AffineLogTransformType, FixedImageType, MovingImageType>;
  using TransformInitializerPointer = typename TransformInitializerType::Pointer;

  /** For scales setting in the optimizer */
  using typename Superclass2::ScalesType;

  /** Execute stuff before the actual registration:
   * \li Call InitializeTransform
   * \li Set the scales.
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

  /** Function to read transform-parameters from a file.
   *
   * It reads the center of rotation and calls the superclass' implementation.
   */
  void
  ReadFromFile() override;

protected:
  /** The constructor. */
  AffineLogTransformElastix();

  /** The destructor. */
  ~AffineLogTransformElastix() override = default;

  /** Try to read the CenterOfRotationPoint from the transform parameter file
   * The CenterOfRotationPoint is already in world coordinates. */
  virtual bool
  ReadCenterOfRotationPoint(InputPointType & rotationPoint) const;

private:
  elxOverrideGetSelfMacro;

  /** Initialize Transform.
   * \li Set all parameters to zero.
   * \li Set center of rotation:
   *  automatically initialized to the geometric center of the image, or
   *   assigned a user entered voxel index, given by the parameter
   *   (CenterOfRotation <index-x> <index-y> ...);
   *   If an initial transform is present and HowToCombineTransforms is
   *   set to "Compose", the initial transform is taken into account
   *   while setting the center of rotation.
   * \li Set initial translation:
   *  the initial translation between fixed and moving image is guessed,
   *  if the user has set (AutomaticTransformInitialization "true").
   *
   * It is not yet possible to enter an initial rotation angle.
   */
  void
  InitializeTransform();

  /** Creates a map of the parameters specific for this (derived) transform type. */
  ParameterMapType
  CreateDerivedTransformParametersMap() const override;

  const AffineLogTransformPointer m_AffineLogTransform{ AffineLogTransformType::New() };
};

} // end namespace elastix

#endif // ELXAFFINELOGTRANSFORM_H

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxAffineLogTransform.hxx"
#endif
