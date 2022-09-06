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
#ifndef elxSimilarityTransform_h
#define elxSimilarityTransform_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkSimilarityTransform.h"
#include "itkCenteredTransformInitializer.h"

namespace elastix
{

/**
 * \class SimilarityTransformElastix
 * \brief A transform based on the itk SimilarityTransforms.
 *
 * This transform is a rigid body transformation, with an isotropic scaling.
 * In 2D, the order of parameters is:\n
 *   [scale, rotation angle, translationx, translationy]\n
 * In 3D, the order of parameters is: \n
 *   [versor1 versor2 versor3 translationx translationy translationz scale]\n
 * Make sure, when specifying the Scales manually that you keep in mind this order!
 *
 * The parameters used in this class are:
 * \parameter Transform: Select this transform as follows:\n
 *    <tt>(%Transform "SimilarityTransform")</tt>
 * \parameter Scales: the scale factor between the rotations, translations,
 *    and the isotropic scaling, used in the optimizer. \n
 *    example: <tt>(Scales 100000.0 60000.0 ... 80000.0)</tt> \n
 *    With this transform, the number of arguments should be
 *    equal to the number of parameters: for each parameter its scale factor.
 *    If this parameter option is not used, by default the rotations are scaled
 *    by a factor of 100000.0 and the scale by a factor 10000.0.
 *    These are rather arbitrary values. See also the AutomaticScalesEstimation parameter.
 *    See also the comment in the documentation of SimilarityTransformElastix about
 *    the order of the parameters in 2D and 3D.
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
 *
 * The transform parameters necessary for transformix, additionally defined by this class, are:
 * \transformparameter CenterOfRotation: stores the center of rotation as an index. \n
 *    example: <tt>(CenterOfRotation 128 128 90)</tt>\n
 *    <b>depecrated!</b> From elastix version 3.402 this is changed to CenterOfRotationPoint!
 * \transformparameter CenterOfRotationPoint: stores the center of rotation, expressed in world coordinates. \n
 *    example: <tt>(CenterOfRotationPoint 10.555 6.666 12.345)</tt>
 *
 * \ingroup Transforms
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT SimilarityTransformElastix
  : public itk::AdvancedCombinationTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                             elx::TransformBase<TElastix>::FixedImageDimension>
  , public elx::TransformBase<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(SimilarityTransformElastix);

  /** Standard ITK-stuff. */
  using Self = SimilarityTransformElastix;
  using Superclass1 = itk::AdvancedCombinationTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                                        elx::TransformBase<TElastix>::FixedImageDimension>;
  using Superclass2 = elx::TransformBase<TElastix>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** The ITK-class that provides most of the functionality, and
   * that is set as the "CurrentTransform" in the CombinationTransform */
  using SimilarityTransformType = itk::SimilarityTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                                           elx::TransformBase<TElastix>::FixedImageDimension>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(SimilarityTransformElastix, itk::AdvancedCombinationTransform);

  /** Name of this class.
   * Use this name in the parameter file to select this specific transform. \n
   * example: <tt>(Transform "SimilarityTransform")</tt>\n
   */
  elxClassNameMacro("SimilarityTransform");

  /** Dimension of the fixed image. */
  itkStaticConstMacro(SpaceDimension, unsigned int, Superclass2::FixedImageDimension);

  /** Typedefs inherited from the superclass. */

  /** These are both in Similarity2D and Similarity3D. */
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

  /** NOTE: use this one only in 3D (otherwise it's just an int). */
  using SimilarityTransformPointer = typename SimilarityTransformType::Pointer;
  using OffsetType = typename SimilarityTransformType::OffsetType;

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
    itk::CenteredTransformInitializer<SimilarityTransformType, FixedImageType, MovingImageType>;
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
  SimilarityTransformElastix();
  /** The destructor. */
  ~SimilarityTransformElastix() override = default;

  /** Try to read the CenterOfRotation from the transform parameter file
   * This is an index value, and, thus, converted to world coordinates.
   * Transform parameter files generated by elastix version < 3.402
   * saved the center of rotation in this way.
   */
  virtual bool
  ReadCenterOfRotationIndex(InputPointType & rotationPoint) const;

  /** Try to read the CenterOfRotationPoint from the transform parameter file
   * The CenterOfRotationPoint is already in world coordinates.
   * Transform parameter files generated by elastix version > 3.402
   * save the center of rotation in this way.
   */
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

  const SimilarityTransformPointer m_SimilarityTransform{ SimilarityTransformType::New() };
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxSimilarityTransform.hxx"
#endif

#endif // end #ifndef elxSimilarityTransform_h
