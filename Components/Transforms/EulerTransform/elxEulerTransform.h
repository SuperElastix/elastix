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
#ifndef elxEulerTransform_h
#define elxEulerTransform_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkAdvancedCombinationTransform.h"
#include "itkEulerTransform.h"
#include "itkCenteredTransformInitializer.h"

namespace elastix
{

/**
 * \class EulerTransformElastix
 * \brief A transform based on the itk EulerTransforms.
 *
 * This transform is a rigid body transformation.
 *
 * The parameters used in this class are:
 * \parameter Transform: Select this transform as follows:\n
 *    <tt>(%Transform "EulerTransform")</tt>
 * \parameter Scales: the scale factor between the rotations and translations,
 *    used in the optimizer. \n
 *    example: <tt>(Scales 200000.0)</tt> \n
 *    example: <tt>(Scales 100000.0 60000.0 ... 80000.0)</tt> \n
 *    If only one argument is given, that factor is used for the rotations.
 *    If more than one argument is given, then the number of arguments should be
 *    equal to the number of parameters: for each parameter its scale factor.
 *    If this parameter option is not used, by default the rotations are scaled
 *    by a factor of 100000.0. See also the AutomaticScalesEstimation parameter.
 * \parameter AutomaticScalesEstimation: if this parameter is set to "true" the Scales
 *    parameter is ignored and the scales are determined automatically. \n
 *    example: <tt>( AutomaticScalesEstimation "true" ) </tt> \n
 *    Default: "false" (for backwards compatibility). Recommended: "true".
 * \parameter CenterOfRotation: an index around which the image is rotated. \n
 *    example: <tt>(CenterOfRotation 128 128 90)</tt> \n
 *    By default the CenterOfRotation is set to the geometric center of the image.
 * \parameter AutomaticTransformInitialization: whether or not the initial translation
 *    between images should be estimated as the distance between their centers.\n
 *    example: <tt>(AutomaticTransformInitialization "true")</tt> \n
 *    By default "false" is assumed. So, no initial translation.
 * \parameter AutomaticTransformInitializationMethod: how to initialize this
 *    transform. Should be one of {GeometricalCenter, CenterOfGravity}.\n
 *    example: <tt>(AutomaticTransformInitializationMethod "CenterOfGravity")</tt> \n
 *    By default "GeometricalCenter" is assumed.\n
 * \parameter ComputeZYX: whether the order of rotations is ZYX. Default: "false". When false, the order is ZXY.\n
 *    This parameter is only relevant for 3D transformation, otherwise it is ignored.\n
 *
 * The transform parameters necessary for transformix, additionally defined by this class, are:
 * \transformparameter CenterOfRotationPoint: stores the center of rotation, expressed in world coordinates. \n
 *    example: <tt>(CenterOfRotationPoint 10.555 6.666 12.345)</tt>
 *
 * \ingroup Transforms
 */

template <typename TElastix>
class ITK_TEMPLATE_EXPORT EulerTransformElastix
  : public itk::AdvancedCombinationTransform<typename elx::TransformBase<TElastix>::CoordinateType,
                                             elx::TransformBase<TElastix>::FixedImageDimension>
  , public elx::TransformBase<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(EulerTransformElastix);

  /** Standard ITK-stuff.*/
  using Self = EulerTransformElastix;

  using Superclass1 = itk::AdvancedCombinationTransform<typename elx::TransformBase<TElastix>::CoordinateType,
                                                        elx::TransformBase<TElastix>::FixedImageDimension>;

  using Superclass2 = elx::TransformBase<TElastix>;

  /** The ITK-class that provides most of the functionality, and
   * that is set as the "CurrentTransform" in the CombinationTransform */
  using EulerTransformType = itk::EulerTransform<typename elx::TransformBase<TElastix>::CoordinateType,
                                                 elx::TransformBase<TElastix>::FixedImageDimension>;

  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  // itkTypeMacro( EulerTransformElastix, EulerTransform );
  itkOverrideGetNameOfClassMacro(EulerTransformElastix);

  /** Name of this class.
   * Use this name in the parameter file to select this specific transform. \n
   * example: <tt>(Transform "EulerTransform")</tt>\n
   */
  elxClassNameMacro("EulerTransform");

  /** Dimension of the fixed image. */
  itkStaticConstMacro(SpaceDimension, unsigned int, Superclass2::FixedImageDimension);

  /** Typedefs inherited from the superclass. */

  /** These are both in Euler2D and Euler3D. */
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

  using EulerTransformPointer = typename EulerTransformType::Pointer;
  using OffsetType = typename EulerTransformType::OffsetType;

  /** Typedef's inherited from TransformBase. */
  using typename Superclass2::ElastixType;
  using typename Superclass2::ParameterMapType;
  using typename Superclass2::RegistrationType;
  using typename Superclass2::CoordinateType;
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
    itk::CenteredTransformInitializer<EulerTransformType, FixedImageType, MovingImageType>;
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
  EulerTransformElastix();
  /** The destructor. */
  ~EulerTransformElastix() override = default;

private:
  elxOverrideGetSelfMacro;

  /** Try to read the CenterOfRotationPoint from the transform parameter file
   * The CenterOfRotationPoint is already in world coordinates.
   * Transform parameter files generated by elastix version > 3.402
   * save the center of rotation in this way.
   */
  bool
  ReadCenterOfRotationPoint(InputPointType & rotationPoint) const;

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
  CreateDerivedTransformParameterMap() const override;

  const EulerTransformPointer m_EulerTransform{ EulerTransformType::New() };
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxEulerTransform.hxx"
#endif

#endif // end #ifndef elxEulerTransform_h
