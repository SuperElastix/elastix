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
#ifndef elxAffineDTITransform_h
#define elxAffineDTITransform_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkAdvancedCombinationTransform.h"
#include "itkAffineDTITransform.h"
#include "itkCenteredTransformInitializer.h"

namespace elastix
{

/**
 * \class AffineDTITransformElastix
 * \brief A transform based on the itk AffineDTITransform.
 *
 * This transform is an affine transformation, with a different parametrisation
 * than the usual one. It is specialised for MR-DTI applications.
 * See also the description of the itk::AffineDTI3DTransform.
 *
 * \warning: the behaviour of this transform might still change in the future. It is still experimental.
 *
 * The parameters used in this class are:
 * \parameter Transform: Select this transform as follows:\n
 *    <tt>(%Transform "AffineDTITransform")</tt>
 * \parameter Scales: the scale factor for each parameter, used by the optimizer. \n
 *    example: <tt>(Scales -1 -1 -1 100000000 -1 -1 -1 -1 -1 -1 -1 -1)</tt> \n
 *    The scales work a bit different in this transformation, compared with the other
 *    transformations like the EulerTransform. By default, if no scales are
 *    supplied, the scales are estimated automatically. With the scales parameter,
 *    the user can override the automatically determined scales. Values smaller than
 *    or equal to 0 are ignored: the automatically estimated value is used for these
 *    parameters. Positive values override the automatically estimated scales. So,
 *    in the example above, the shear_x parameter is given a very large scale, which
 *    effectively turns off the optimization of that parameters. This might be useful
 *    in some MR-DTI application.
 * \parameter CenterOfRotation: an index around which the image is rotated. \n
 *    example: <tt>(CenterOfRotation 128 128 90)</tt> \n
 *    By default the CenterOfRotation is set to the geometric center of the image.
 * \parameter AutomaticTransformInitialization: whether or not the initial translation
 *    between images should be estimated as the distance between their centers.\n
 *    example: <tt>(AutomaticTransformInitialization "true")</tt> \n
 *    By default "false" is assumed. So, no initial translation.
 *
 * The transform parameters necessary for transformix, additionally defined by this class, are:   *
 * \transformparameter CenterOfRotation: stores the center of rotation as an index. \n
 *    example: <tt>(CenterOfRotation 128 128 90)</tt>\n
 *    <b>depecrated!</b> From elastix version 3.402 this is changed to CenterOfRotationPoint!
 * \transformparameter CenterOfRotationPoint: stores the center of rotation, expressed in world coordinates. \n
 *    example: <tt>(CenterOfRotationPoint 10.555 6.666 12.345)</tt>
 * \transformparameter MatrixTranslation: the parameters as matrix and translation, so written
 *    in the same way as the parameters of the normal AffineTransform.\n
 *    example: <tt>(MatrixTranslation 1 0 0 0 1 0 0 0 0.9 0.1 0.1 0.2)</tt>\n
 *    This parameter is just for convenience, so that the user can compare the results between
 *    the AffineTransform and AffineDTITransform. It is not a mandatory parameter.
 *
 * \ingroup Transforms
 * \sa AffineDTI3DTransform
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT AffineDTITransformElastix
  : public itk::AdvancedCombinationTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                             elx::TransformBase<TElastix>::FixedImageDimension>
  , public elx::TransformBase<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(AffineDTITransformElastix);

  /** Standard ITK-stuff.*/
  using Self = AffineDTITransformElastix;
  using Superclass1 = itk::AdvancedCombinationTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                                        elx::TransformBase<TElastix>::FixedImageDimension>;
  using Superclass2 = elx::TransformBase<TElastix>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** The ITK-class that provides most of the functionality, and
   * that is set as the "CurrentTransform" in the CombinationTransform */
  using AffineDTITransformType = itk::AffineDTITransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                                         elx::TransformBase<TElastix>::FixedImageDimension>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(AffineDTITransformElastix, AdvancedCombinationTransform);

  /** Name of this class.
   * Use this name in the parameter file to select this specific transform. \n
   * example: <tt>(Transform "AffineDTITransform")</tt>\n
   */
  elxClassNameMacro("AffineDTITransform");

  /** Dimension of the fixed image. */
  itkStaticConstMacro(SpaceDimension, unsigned int, Superclass2::FixedImageDimension);

  /** Typedefs inherited from the superclass. */

  /** These are both in AffineDTI2D and AffineDTI3D. */
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

  using AffineDTITransformPointer = typename AffineDTITransformType::Pointer;
  using OffsetType = typename AffineDTITransformType::OffsetType;

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
    itk::CenteredTransformInitializer<AffineDTITransformType, FixedImageType, MovingImageType>;
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
  AffineDTITransformElastix();
  /** The destructor. */
  ~AffineDTITransformElastix() override = default;

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

  const AffineDTITransformPointer m_AffineDTITransform{ AffineDTITransformType::New() };
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxAffineDTITransform.hxx"
#endif

#endif // end #ifndef elxAffineDTITransform_h
