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
#ifndef __elxAdvancedLimitedEulerTransform_H__
#define __elxAdvancedLimitedEulerTransform_H__

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkAdvancedCombinationTransform.h"
#include "itkLimitedEulerTransform.h"
#include "itkCenteredTransformInitializer.h"

namespace elastix
{

/**
 * \class AdvancedLimitedEulerTransformElastix
 * \brief A transform based on the itk LimitedEulerTransforms.
 *
 * This transform is a rigid body transformation.
 *
 * The parameters used in this class are:
 * \parameter Transform: Select this transform as follows:\n
 *    <tt>(%Transform "LimitedEulerTransform")</tt>
 * \parameter UpperLimits: the upper limit value for the rotations and translations,
 *    used in the optimizer. \n
 *    example: <tt>(UpperLimits 3.14 3.14 3.14 100.00 100.00 100.00)</tt> \n
 *    The number of arguments should be equal to the number of parameters: 
 *    for each parameter its upper limit value. Arguments are in the order as for
 *    the Euler transform parameters: <RotX RotY RotZ TraX TraY TraZ>. Set the limit 
 *    value to extreme positive values to (e.g. 1e+3) to effectively disable the upper limits.
 * \parameter LowerLimits: the lower limit values for the rotations and translations,
 *    used in the optimizer. \n
 *    example: <tt>(LowerLimits -3.14 -3.14 -3.14 -100.00 -100.00 -100.00)</tt> \n
 *    The number of arguments should be equal to the number of parameters: 
 *    for each parameter its lower limit value. Arguments are in the order as for
 *    the Euler transform parameters: <RotX RotY RotZ TraX TraY TraZ>. Set the limit 
 *    value to extreme negative values to (e.g. -1e+3) to effectively disable the upper limits.
 * \parameter SharpnessOfLimits: the scale factor which determines the limit cutoff sharpness. \n
 *    example: <tt>(LimitSharpness 25.0)</tt> \n
 *    High values impose a sharp limit (e.g. 10.0--100.0), while lower values (1.0-10.0). 
 *    Minimum value is 1.0, lower values will be clamped to 1.0. If this parameter option 
 *    is not used, by default for the sharpness limit value is 25.0.* 
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

template< class TElastix >
class AdvancedLimitedEulerTransformElastix :
  public itk::AdvancedCombinationTransform<
  typename elx::TransformBase< TElastix >::CoordRepType,
  elx::TransformBase< TElastix >::FixedImageDimension >,
  public elx::TransformBase< TElastix >
{
public:

  /** Standard ITK-stuff.*/
  typedef AdvancedLimitedEulerTransformElastix Self;

  typedef itk::AdvancedCombinationTransform<
    typename elx::TransformBase< TElastix >::CoordRepType,
    elx::TransformBase< TElastix >::FixedImageDimension >     Superclass1;

  typedef elx::TransformBase< TElastix > Superclass2;

  /** The ITK-class that provides most of the functionality, and
   * that is set as the "CurrentTransform" in the CombinationTransform */
  typedef itk::LimitedEulerTransform<
    typename elx::TransformBase< TElastix >::CoordRepType,
    elx::TransformBase< TElastix >::FixedImageDimension >     LimitedEulerTransformType;

  // typedef itk::LimitedEuler3DTransform<
  //   typename elx::TransformBase< TElastix >::CoordRepType >     LimitedEulerTransformType;

  typedef itk::SmartPointer< Self >       Pointer;
  typedef itk::SmartPointer< const Self > ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  //itkTypeMacro( AdvancedLimitedEulerTransformElastix, LimitedEulerTransform );
  itkTypeMacro( AdvancedLimitedEulerTransformElastix, itk::AdvancedCombinationTransform );

  /** Name of this class.
   * Use this name in the parameter file to select this specific transform. \n
   * example: <tt>(Transform "AdvancedLimitedEulerTransform")</tt>\n
   */
  elxClassNameMacro( "AdvancedLimitedEulerTransform" );

  /** Dimension of the fixed image. */
  itkStaticConstMacro( SpaceDimension, unsigned int, Superclass2::FixedImageDimension );

  /** Typedefs inherited from the superclass. */

  /** These are both in Euler2D and Euler3D. */
  typedef typename Superclass1::ScalarType             ScalarType;
  typedef typename Superclass1::ParametersType         ParametersType;
  typedef typename Superclass1::NumberOfParametersType NumberOfParametersType;
  typedef typename Superclass1::JacobianType           JacobianType;

  typedef typename Superclass1::InputPointType            InputPointType;
  typedef typename Superclass1::OutputPointType           OutputPointType;
  typedef typename Superclass1::InputVectorType           InputVectorType;
  typedef typename Superclass1::OutputVectorType          OutputVectorType;
  typedef typename Superclass1::InputCovariantVectorType  InputCovariantVectorType;
  typedef typename Superclass1::OutputCovariantVectorType OutputCovariantVectorType;
  typedef typename Superclass1::InputVnlVectorType        InputVnlVectorType;
  typedef typename Superclass1::OutputVnlVectorType       OutputVnlVectorType;

  typedef typename LimitedEulerTransformType::Pointer    LimitedEulerTransformPointer;
  typedef typename LimitedEulerTransformType::OffsetType OffsetType;

  /** Typedef's inherited from TransformBase. */
  typedef typename Superclass2::ElastixType              ElastixType;
  typedef typename Superclass2::ElastixPointer           ElastixPointer;
  typedef typename Superclass2::ParameterMapType         ParameterMapType;
  typedef typename Superclass2::ConfigurationType        ConfigurationType;
  typedef typename Superclass2::ConfigurationPointer     ConfigurationPointer;
  typedef typename Superclass2::RegistrationType         RegistrationType;
  typedef typename Superclass2::RegistrationPointer      RegistrationPointer;
  typedef typename Superclass2::CoordRepType             CoordRepType;
  typedef typename Superclass2::FixedImageType           FixedImageType;
  typedef typename Superclass2::MovingImageType          MovingImageType;
  typedef typename Superclass2::ITKBaseType              ITKBaseType;
  typedef typename Superclass2::CombinationTransformType CombinationTransformType;

  /** Other typedef's. */
  typedef typename FixedImageType::IndexType     IndexType;
  typedef typename IndexType::IndexValueType     IndexValueType;
  typedef typename FixedImageType::SizeType      SizeType;
  typedef typename FixedImageType::PointType     PointType;
  typedef typename FixedImageType::SpacingType   SpacingType;
  typedef typename FixedImageType::RegionType    RegionType;
  typedef typename FixedImageType::DirectionType DirectionType;

  typedef itk::CenteredTransformInitializer<
    LimitedEulerTransformType, FixedImageType, MovingImageType >  TransformInitializerType;
  typedef typename TransformInitializerType::Pointer TransformInitializerPointer;

  /** For scales setting in the optimizer */
  typedef typename Superclass2::ScalesType ScalesType;

  /** Execute stuff before the actual registration:
   * \li Call InitializeTransform
   * \li Set the scales.
   */
  void BeforeRegistration( void ) override;

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
  virtual void InitializeTransform( void );

  /** Set the scales
   * \li If AutomaticScalesEstimation is "true" estimate scales
   * \li If scales are provided by the user use those,
   * \li Otherwise use some default value
   * This function is called by BeforeRegistration, after
   * the InitializeTransform function is called
   */
  virtual void SetScales( void );

  /** Function to read transform-parameters from a file.
   *
   * It reads the center of rotation and calls the superclass' implementation.
   */
  void ReadFromFile( void ) override;

  /** Function to create transform-parameters map.
   * Creates the TransformParametersmap
   */
  ParameterMapType
  CreateDerivedTransformParametersMap(void) const override;

  void
  AfterRegistration(void) override;

protected:

  /** The constructor. */
  AdvancedLimitedEulerTransformElastix();
  /** The destructor. */
  ~AdvancedLimitedEulerTransformElastix() override {}

  /** Try to read the CenterOfRotationPoint from the transform parameter file
   * The CenterOfRotationPoint is already in world coordinates.
   * Transform parameter files generated by elastix version > 3.402
   * save the center of rotation in this way.
   */
  virtual bool ReadCenterOfRotationPoint( InputPointType & rotationPoint ) const;

private:
  elxOverrideGetSelfMacro;

  /** The private constructor. */
  AdvancedLimitedEulerTransformElastix( const Self & );  // purposely not implemented
  /** The private copy constructor. */
  void operator=( const Self & );         // purposely not implemented

  LimitedEulerTransformPointer m_LimitedEulerTransform;

};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxAdvancedLimitedEulerTransform.hxx"
#endif

#endif // end #ifndef __elxAdvancedLimitedEulerTransform_H__
