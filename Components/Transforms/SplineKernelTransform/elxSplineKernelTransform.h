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
#ifndef elxSplineKernelTransform_h
#define elxSplineKernelTransform_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkKernelTransform2.h"
#include "itkElasticBodySplineKernelTransform2.h"
#include "itkElasticBodyReciprocalSplineKernelTransform2.h"
#include "itkThinPlateSplineKernelTransform2.h"
#include "itkThinPlateR2LogRSplineKernelTransform2.h"
#include "itkVolumeSplineKernelTransform2.h"

namespace elastix
{

/**
 * \class SplineKernelTransform
 * \brief A transform based on the itk::KernelTransform2.
 *
 * This transform is a nonrigid transformation, based on
 * thin-plate-spline-like kernels.
 *
 * The ITK code for this class is largely based on code by
 * Rupert Brooks. For elastix a few modifications were made
 * (making the transform thread safe, and make it support the
 * AdvancedTransform framework).
 *
 * This nonrigid transformation model allows the user to place control points
 * at application-specific positions, unlike the BSplineTransform, which always
 * uses a regular grid of control points.
 *
 * NB: in order to use this class for registration, the -fp command line
 * argument is mandatory! It is used to place the fixed image landmarks.
 *
 * The parameters used in this class are:
 * \parameter Transform: Select this transform as follows:\n
 *    <tt>(%Transform "SplineKernelTransform")</tt>
 * \parameter SplineKernelType: Select the deformation model, which must
 * be one of { ThinPlateSpline, ThinPlateR2LogRSpline, VolumeSpline,
 * ElasticBodySpline, ElasticBodyReciprocalSpline). In 2D this option is
 * ignored and a ThinPlateSpline will always be used. \n
 *   example: <tt>(SplineKernelType "ElasticBodySpline")</tt>\n
 * Default: ThinPlateSpline. You cannot specify this parameter for each
 * resolution differently.
 * \parameter SplineRelaxationFactor: make the spline interpolating or
 * approximating. A value of 0.0 gives an interpolating transform. Higher
 * values result in approximating splines.\n
 *   example: <tt>(SplineRelaxationFactor 0.01 )</tt>\n
 * Default: 0.0. You cannot specify this parameter for each resolution differently.
 * \parameter SplinePoissonRatio: Set the poisson ratio for the
 * ElasticBodySpline and the ElastixBodyReciprocalSpline. For other
 * SplineKernelTypes this parameters is ignored.\n
 *   example: <tt>(SplinePoissonRatio 0.3 )</tt>\n
 * Default: 0.3. You cannot specify this parameter for each resolution differently.\n
 * Valid values are withing -1.0 and 0.5. 0.5 means incompressible.
 * Negative values are a bit odd, but possible. See Wikipedia on PoissonRatio.
 *
 * \commandlinearg -fp: a file specifying a set of points that will serve
 * as fixed image landmarks.\n
 *   example: <tt>-fp fixedImagePoints.txt</tt> \n
 *   The fixedImagePoints.txt file should be structured: first line should
 * be "index" or "point", depending if the user supplies voxel indices or
 * real world coordinates. The second line should be the number of points
 * that should be transformed. The third and following lines give the
 * indices or points. The same structure thus as used for transformix.\n
 * \commandlinearg -mp: an optional file specifying a set of points that will serve
 * as moving image landmarks, used to initialize the transformation.\n
 *   example: <tt>-mp movingImagePoints.txt</tt> \n
 *   The movingImagePoints.txt should be structured like the fixedImagePoints.txt.
 *  The moving landmarks should be corresponding to the fixed landmarks.
 *  If no file is provided, the transformation is initialized to be the identity,
 *  i.e. the moving landmarks are chosen identical to the fixed landmarks.
 *
 * \transformparameter Transform: Select this transform as follows:\n
 *    <tt>(%Transform "SplineKernelTransform")</tt>
 * \transformparameter SplineKernelType: Select the deformation model,
 * which must be one of { ThinPlateSpline, ThinPlateR2LogRSpline, VolumeSpline,
 * ElasticBodySpline, ElasticBodyReciprocalSpline). In 2D this option is
 * ignored and a ThinPlateSpline will always be used. \n
 *   example: <tt>(SplineKernelType "ElasticBodySpline")</tt>\n   *
 * \transformparameter SplineRelaxationFactor: make the spline interpolating
 * or approximating. A value of 0.0 gives an interpolating transform.
 * Higher values result in approximating splines.\n
 *   example: <tt>(SplineRelaxationFactor 0.01 )</tt>\n   *
 * \transformparameter SplinePoissonRatio: Set the Poisson ratio for the
 * ElasticBodySpline and the ElastixBodyReciprocalSpline. For other
 * SplineKernelTypes this parameters is ignored.\n
 *   example: <tt>(SplinePoissonRatio 0.3 )</tt>\n
 * Valid values are withing -1.0 and 0.5. 0.5 means incompressible.
 * Negative values are a bit odd, but possible. See Wikipedia on PoissonRatio.
 * \transformparameter FixedImageLandmarks: The landmark positions in the
 * fixed image, in world coordinates. Positions written as x1 y1 [z1] x2 y2 [z2] etc.\n
 *   example: <tt>(FixedImageLandmarks 10.0 11.0 12.0 4.0 4.0 4.0 6.0 6.0 6.0 )</tt>
 *
 * \ingroup Transforms
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT SplineKernelTransform
  : public itk::AdvancedCombinationTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                             elx::TransformBase<TElastix>::FixedImageDimension>
  , public elx::TransformBase<TElastix>
{
public:
  /** Standard ITK-stuff. */
  typedef SplineKernelTransform Self;
  typedef itk::AdvancedCombinationTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                            elx::TransformBase<TElastix>::FixedImageDimension>
                                       Superclass1;
  typedef elx::TransformBase<TElastix> Superclass2;

  /** The ITK-class that provides most of the functionality, and
   * that is set as the "CurrentTransform" in the CombinationTransform.
   */
  typedef itk::KernelTransform2<typename elx::TransformBase<TElastix>::CoordRepType,
                                elx::TransformBase<TElastix>::FixedImageDimension>
                                        KernelTransformType;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(SplineKernelTransform, itk::AdvancedCombinationTransform);

  /** Name of this class.
   * Use this name in the parameter file to select this specific transform. \n
   * example: <tt>(Transform "SplineKernelTransform")</tt>\n
   */
  elxClassNameMacro("SplineKernelTransform");

  /** Dimension of the domain space. */
  itkStaticConstMacro(SpaceDimension, unsigned int, Superclass2::FixedImageDimension);

  /** Typedefs inherited from the superclass. */
  typedef typename Superclass1::ScalarType                ScalarType;
  typedef typename Superclass1::ParametersType            ParametersType;
  typedef typename Superclass1::NumberOfParametersType    NumberOfParametersType;
  typedef typename Superclass1::JacobianType              JacobianType;
  typedef typename Superclass1::InputVectorType           InputVectorType;
  typedef typename Superclass1::OutputVectorType          OutputVectorType;
  typedef typename Superclass1::InputCovariantVectorType  InputCovariantVectorType;
  typedef typename Superclass1::OutputCovariantVectorType OutputCovariantVectorType;
  typedef typename Superclass1::InputVnlVectorType        InputVnlVectorType;
  typedef typename Superclass1::OutputVnlVectorType       OutputVnlVectorType;
  typedef typename Superclass1::InputPointType            InputPointType;
  typedef typename Superclass1::OutputPointType           OutputPointType;

  /** Typedef's from the TransformBase class. */
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

  /** Extra typedefs */
  typedef typename KernelTransformType::Pointer      KernelTransformPointer;
  typedef typename KernelTransformType::PointSetType PointSetType;
  typedef typename PointSetType::Pointer             PointSetPointer;

  /** Execute stuff before everything else:
   * \li Check if -fp command line argument was given
   * \li Check if -mp command line argument was given
   */
  int
  BeforeAll(void) override;

  /** Execute stuff before the actual registration:
   * \li Setup transform
   * \li Determine fixed image (source) landmarks
   * \li Determine moving image (target) landmarks
   * \li Call InitializeTransform.
   */
  void
  BeforeRegistration(void) override;

  /** Function to read transform-parameters from a file. */
  void
  ReadFromFile(void) override;

protected:
  /** The constructor. */
  SplineKernelTransform();
  /** The destructor. */
  ~SplineKernelTransform() override = default;

  typedef itk::ThinPlateSplineKernelTransform2<CoordRepType, itkGetStaticConstMacro(SpaceDimension)>
    TPKernelTransformType;
  typedef itk::ThinPlateR2LogRSplineKernelTransform2<CoordRepType, itkGetStaticConstMacro(SpaceDimension)>
                                                                                                  TPRKernelTransformType;
  typedef itk::VolumeSplineKernelTransform2<CoordRepType, itkGetStaticConstMacro(SpaceDimension)> VKernelTransformType;
  typedef itk::ElasticBodySplineKernelTransform2<CoordRepType, itkGetStaticConstMacro(SpaceDimension)>
    EBKernelTransformType;
  typedef itk::ElasticBodyReciprocalSplineKernelTransform2<CoordRepType, itkGetStaticConstMacro(SpaceDimension)>
    EBRKernelTransformType;

  /** Create an instance of a kernel transform. Returns false if the
   * kernelType is unknown.
   */
  virtual bool
  SetKernelType(const std::string & kernelType);

  /** Read source landmarks from fp file
   * \li Try reading -fp file
   */
  virtual void
  DetermineSourceLandmarks(void);

  /** Read target landmarks from mp file or load identity.
   * \li Try reading -mp file
   * \li If no -mp file was given, place landmarks as identity.
   */
  virtual bool
  DetermineTargetLandmarks(void);

  /** General function to read all landmarks. */
  virtual void
  ReadLandmarkFile(const std::string & filename,
                   PointSetPointer &   landmarkPointSet,
                   const bool &        landmarksInFixedImage);

  /** The itk kernel transform. */
  KernelTransformPointer m_KernelTransform;

private:
  elxOverrideGetSelfMacro;

  /** Creates a map of the parameters specific for this (derived) transform type. */
  ParameterMapType
  CreateDerivedTransformParametersMap(void) const override;

  /** The deleted copy constructor. */
  SplineKernelTransform(const Self &) = delete;
  /** The deleted assignment operator. */
  void
  operator=(const Self &) = delete;

  std::string m_SplineKernelType;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxSplineKernelTransform.hxx"
#endif

#endif // end #ifndef elxSplineKernelTransform_h
