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
#ifndef elxEulerStackTransform_h
#define elxEulerStackTransform_h

#include "elxIncludes.h"
/** Include itk transforms needed. */
#include "itkAdvancedCombinationTransform.h"
#include "itkStackTransform.h"
#include "itkEulerTransform.h"

namespace elastix
{

/**
 * \class EulerStackTransform
 * \brief A stack transform based on the itk EulerTransforms.
 *
 * This transform is a rigid body transformation. Calls to TransformPoint and GetJacobian are
 * redirected to the appropriate sub transform based on the last dimension (time) index.
 *
 * This transform uses the size, spacing and origin of the last dimension of the fixed
 * image to set the number of sub transforms, the origin of the first transform and the
 * spacing between the transforms.
 *
 *
 * The parameters used in this class are:
 * \parameter Transform: Select this transform as follows:\n
 *    <tt>(%Transform "EulerStackTransform")</tt>
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
 *    example: <tt>(CenterOfRotation 128 128)</tt> \n
 *
 * The transform parameters necessary for transformix, additionally defined by this class, are:
 * \transformparameter CenterOfRotation: stores the center of rotation as an index. \n
 *    example: <tt>(CenterOfRotation 128 128)</tt>
 *    deprecated! From elastix version 3.402 this is changed to CenterOfRotationPoint!
 * \transformparameter CenterOfRotationPoint: stores the center of rotation, expressed in world coordinates. \n
 *    example: <tt>(CenterOfRotationPoint 10.555 6.666)</tt>
 * \transformparameter StackSpacing: stores the spacing between the sub transforms. \n
 *    exanoke: <tt>(StackSpacing 1.0)</tt>
 * \transformparameter StackOrigin: stores the origin of the first sub transform. \n
 *    exanoke: <tt>(StackOrigin 0.0)</tt>
 * \transformparameter NumberOfSubTransforms: stores the number of sub transforms. \n
 *    exanoke: <tt>(NumberOfSubTransforms 10)</tt>
 *
 * \todo It is unsure what happens when one of the image dimensions has length 1.
 * \todo The center of rotation point is not transformed with the initial transform yet.
 *
 * \ingroup Transforms
 */

template <class TElastix>
class EulerStackTransform
  : public itk::AdvancedCombinationTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                             elx::TransformBase<TElastix>::FixedImageDimension>
  , public elx::TransformBase<TElastix>
{
public:
  /** Standard ITK-stuff. */
  typedef EulerStackTransform Self;
  typedef itk::AdvancedCombinationTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                            elx::TransformBase<TElastix>::FixedImageDimension>
                                        Superclass1;
  typedef elx::TransformBase<TElastix>  Superclass2;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(EulerStackTransform, itk::AdvancedCombinationTransform);

  /** Name of this class.
   * Use this name in the parameter file to select this specific transform. \n
   * example: <tt>(Transform "EulerStackTransform")</tt>\n
   */
  elxClassNameMacro("EulerStackTransform");

  /** (Reduced) dimension of the fixed image. */
  itkStaticConstMacro(SpaceDimension, unsigned int, Superclass2::FixedImageDimension);
  itkStaticConstMacro(ReducedSpaceDimension, unsigned int, Superclass2::FixedImageDimension - 1);

  /** The ITK-class that provides most of the functionality, and
   * that is set as the "CurrentTransform" in the CombinationTransform.
   */
  typedef itk::EulerTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                              itkGetStaticConstMacro(SpaceDimension)>
                                                      EulerTransformType;
  typedef typename EulerTransformType::Pointer        EulerTransformPointer;
  typedef typename EulerTransformType::InputPointType InputPointType;

  /** The ITK-class for the sub transforms, which have a reduced dimension. */
  typedef itk::EulerTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                              itkGetStaticConstMacro(ReducedSpaceDimension)>
                                                               ReducedDimensionEulerTransformType;
  typedef typename ReducedDimensionEulerTransformType::Pointer ReducedDimensionEulerTransformPointer;

  typedef typename ReducedDimensionEulerTransformType::OutputVectorType ReducedDimensionOutputVectorType;
  typedef typename ReducedDimensionEulerTransformType::InputPointType   ReducedDimensionInputPointType;

  /** Typedef for stack transform. */
  typedef itk::StackTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                              itkGetStaticConstMacro(SpaceDimension),
                              itkGetStaticConstMacro(SpaceDimension)>
                                                    EulerStackTransformType;
  typedef typename EulerStackTransformType::Pointer EulerStackTransformPointer;

  /** Typedefs inherited from the superclass. */
  typedef typename Superclass1::ParametersType         ParametersType;
  typedef typename Superclass1::NumberOfParametersType NumberOfParametersType;

  /** Typedef's from TransformBase. */
  typedef typename Superclass2::ElastixType              ElastixType;
  typedef typename Superclass2::ElastixPointer           ElastixPointer;
  typedef typename Superclass2::ConfigurationType        ConfigurationType;
  typedef typename Superclass2::ConfigurationPointer     ConfigurationPointer;
  typedef typename Superclass2::RegistrationType         RegistrationType;
  typedef typename Superclass2::RegistrationPointer      RegistrationPointer;
  typedef typename Superclass2::CoordRepType             CoordRepType;
  typedef typename Superclass2::FixedImageType           FixedImageType;
  typedef typename Superclass2::MovingImageType          MovingImageType;
  typedef typename Superclass2::ITKBaseType              ITKBaseType;
  typedef typename Superclass2::CombinationTransformType CombinationTransformType;

  /** Reduced Dimension typedef's. */
  typedef float                                                                PixelType;
  typedef itk::Image<PixelType, itkGetStaticConstMacro(ReducedSpaceDimension)> ReducedDimensionImageType;
  typedef itk::ImageRegion<itkGetStaticConstMacro(ReducedSpaceDimension)>      ReducedDimensionRegionType;
  typedef typename ReducedDimensionImageType::PointType                        ReducedDimensionPointType;
  typedef typename ReducedDimensionImageType::SizeType                         ReducedDimensionSizeType;
  typedef typename ReducedDimensionRegionType::IndexType                       ReducedDimensionIndexType;
  typedef typename ReducedDimensionImageType::SpacingType                      ReducedDimensionSpacingType;
  typedef typename ReducedDimensionImageType::DirectionType                    ReducedDimensionDirectionType;
  typedef typename ReducedDimensionImageType::PointType                        ReducedDimensionOriginType;

  /** For scales setting in the optimizer */
  typedef typename Superclass2::ScalesType ScalesType;

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
  BeforeAll(void) override;

  /** Execute stuff before the actual registration:
   * \li Set the stack transform parameters.
   * \li Set initial sub transforms.
   * \li Create initial registration parameters.
   */
  void
  BeforeRegistration(void) override;

  /** Method initialize the parameters (to 0). */
  virtual void
  InitializeTransform(void);

  /** Set the scales
   * \li If AutomaticScalesEstimation is "true" estimate scales
   * \li If scales are provided by the user use those,
   * \li Otherwise use some default value
   * This function is called by BeforeRegistration, after
   * the InitializeTransform function is called
   */
  virtual void
  SetScales(void);

  /** Function to read transform-parameters from a file. */
  void
  ReadFromFile(void) override;

  /** Function to write transform-parameters to a file. */
  void
  WriteToFile(const ParametersType & param) const override;

  /** Function to rotate center of rotation point using initial transformation. */
  virtual void
  InitialTransformCenter(ReducedDimensionInputPointType & point);

protected:
  /** The constructor. */
  EulerStackTransform();

  /** The destructor. */
  ~EulerStackTransform() override {}

  /** Try to read the CenterOfRotationPoint from the transform parameter file
   * The CenterOfRotationPoint is already in world coordinates.
   * Transform parameter files generated by elastix version > 3.402
   * save the center of rotation in this way.
   */
  virtual bool
  ReadCenterOfRotationPoint(ReducedDimensionInputPointType & rotationPoint) const;

private:
  /** The deleted copy constructor and assignment operator. */
  EulerStackTransform(const Self &) = delete;
  void
  operator=(const Self &) = delete;

  /** The Affine stack transform. */
  EulerStackTransformPointer m_EulerStackTransform;

  /** Dummy sub transform to be used to set sub transforms of stack transform. */
  ReducedDimensionEulerTransformPointer m_EulerDummySubTransform;

  /** Stack variables. */
  unsigned int m_NumberOfSubTransforms;
  double       m_StackOrigin, m_StackSpacing;

  /** Initialize the affine transform. */
  unsigned int
  InitializeEulerTransform();
};


} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxEulerStackTransform.hxx"
#endif

#endif // end #ifndef elxEulerStackTransform_h
