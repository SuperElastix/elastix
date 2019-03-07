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
#ifndef __elxAdaptiveStochasticLBFGS_h
#define __elxAdaptiveStochasticLBFGS_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkAdaptiveStochasticLBFGSOptimizer.h"

#include "elxProgressCommand.h"
#include "itkAdvancedTransform.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"
#include "itkComputeJacobianTerms.h"
#include "itkComputeDisplacementDistribution.h"
#include "itkMultiThreader.h"
#include "itkImageRandomSampler.h"
#include "itkLineSearchOptimizer.h"
#include "itkMoreThuenteLineSearchOptimizer.h"


namespace elastix
{
 /**
  * \class AdaptiveStochasticLBFGS
  * \brief A gradient descent optimizer with an adaptive gain.
  *
  * This class is a wrap around the AdaptiveStochasticVarianceReducedGradientOptimizer class.
  * It takes care of setting parameters and printing progress information.
  * For more information about the optimization method, please read the documentation
  * of the AdaptiveStochasticVarianceReducedGradientOptimizer class.
  *
  * This optimizer is very suitable to be used in combination with the Random image sampler,
  * or with the RandomCoordinate image sampler, with the setting (NewSamplesEveryIteration "true").
  * Much effort has been spent on providing reasonable default values for all parameters, to
  * simplify usage. In most registration problems, good results should be obtained without specifying
  * any of the parameters described below (except the first of course, which defines the optimizer
  * to use).
  *
  * This optimization method is described in the following references:
  *
  * [1] P. Cruz,
  * "Almost sure convergence and asymptotical normality of a generalization of Kesten's
  * stochastic approximation algorithm for multidimensional case."
  * Technical Report, 2005. http://hdl.handle.net/2052/74
  *
  * [2] S. Klein, J.P.W. Pluim, and M. Staring, M.A. Viergever,
  * "Adaptive stochastic gradient descent optimisation for image registration,"
  * International Journal of Computer Vision, vol. 81, no. 3, pp. 227-239, 2009.
  * http://dx.doi.org/10.1007/s11263-008-0168-y
  *
  * Acceleration in case of many transform parameters was proposed in the following paper:
  *
  * [3]  Y.Qiao, B.P.F. Lelieveldt, M.Staring
  * "Fast automatic estimation of the optimization step size for nonrigid image registration,"
  * SPIE Medical Imaging: Image Processing,February, 2014.
  * http://elastix.isi.uu.nl/marius/publications/2014_c_SPIEMI.php
  *
  * The parameters used in this class are:
  * \parameter Optimizer: Select this optimizer as follows:\n
  *   <tt>(Optimizer "AdaptiveStochasticLBFGS")</tt>
  * \parameter MaximumNumberOfIterations: The maximum number of iterations in each resolution. \n
  *   example: <tt>(MaximumNumberOfIterations 100 100 50)</tt> \n
  *    Default/recommended value: 500. When you are in a hurry, you may go down to 250 for example.
  *    When you have plenty of time, and want to be absolutely sure of the best results, a setting
  *    of 2000 is reasonable. In general, 500 gives satisfactory results.
  * \parameter MaximumNumberOfSamplingAttempts: The maximum number of sampling attempts. Sometimes
  *   not enough corresponding samples can be drawn, upon which an exception is thrown. With this
  *   parameter it is possible to try to draw another set of samples. \n
  *   example: <tt>(MaximumNumberOfSamplingAttempts 10 15 10)</tt> \n
  *    Default value: 0, i.e. just fail immediately, for backward compatibility.
  * \parameter AutomaticParameterEstimation: When this parameter is set to "true",
  *   many other parameters are calculated automatically: SP_a, SP_alpha, SigmoidMax,
  *   SigmoidMin, and SigmoidScale. In the elastix.log file the actually chosen values for
  *   these parameters can be found. \n
  *   example: <tt>(AutomaticParameterEstimation "true")</tt>\n
  *   Default/recommended value: "true". The parameter can be specified for each resolution,
  *   or for all resolutions at once.
  * \parameter UseAdaptiveStepSizes: When this parameter is set to "true", the adaptive
  *   step size mechanism described in the documentation of
  *   itk::AdaptiveStochasticVarianceReducedGradientOptimizer is used.
  *   The parameter can be specified for each resolution, or for all resolutions at once.\n
  *   example: <tt>(UseAdaptiveStepSizes "true")</tt>\n
  *   Default/recommend value: "true", because it makes the registration more robust. In case
  *   of using a RandomCoordinate sampler, with (UseRandomSampleRegion "true"), the adaptive
  *   step size mechanism is turned off, no matter the user setting.
  * \parameter MaximumStepLength: Also called \f$\delta\f$. This parameter can be considered as
  *   the maximum voxel displacement between two iterations. The larger this parameter, the
  *   more agressive the optimization.
  *   The parameter can be specified for each resolution, or for all resolutions at once.\n
  *   example: <tt>(MaximumStepLength 1.0)</tt>\n
  *   Default: mean voxel spacing of fixed and moving image. This seems to work well in general.
  *   This parameter only has influence when AutomaticParameterEstimation is used.
  * \parameter SP_a: The gain \f$a(k)\f$ at each iteration \f$k\f$ is defined by \n
  *   \f$a(k) =  SP\_a / (SP\_A + k + 1)^{SP\_alpha}\f$. \n
  *   SP_a can be defined for each resolution. \n
  *   example: <tt>(SP_a 3200.0 3200.0 1600.0)</tt> \n
  *   The default value is 400.0. Tuning this variable for you specific problem is recommended.
  *   Alternatively set the AutomaticParameterEstimation to "true". In that case, you do not
  *   need to specify SP_a. SP_a has no influence when AutomaticParameterEstimation is used.
  * \parameter SP_A: The gain \f$a(k)\f$ at each iteration \f$k\f$ is defined by \n
  *   \f$a(k) =  SP\_a / (SP\_A + k + 1)^{SP\_alpha}\f$. \n
  *   SP_A can be defined for each resolution. \n
  *   example: <tt>(SP_A 50.0 50.0 100.0)</tt> \n
  *   The default/recommended value for this particular optimizer is 20.0.
  * \parameter SP_alpha: The gain \f$a(k)\f$ at each iteration \f$k\f$ is defined by \n
  *   \f$a(k) =  SP\_a / (SP\_A + k + 1)^{SP\_alpha}\f$. \n
  *   SP_alpha can be defined for each resolution. \n
  *   example: <tt>(SP_alpha 0.602 0.602 0.602)</tt> \n
  *   The default/recommended value for this particular optimizer is 1.0.
  *   Alternatively set the AutomaticParameterEstimation to "true". In that case, you do not
  *   need to specify SP_alpha. SP_alpha has no influence when AutomaticParameterEstimation is used.
  * \parameter SigmoidMax: The maximum of the sigmoid function (\f$f_{max}\f$). Must be larger than 0.
  *   The parameter can be specified for each resolution, or for all resolutions at once.\n
  *   example: <tt>(SigmoidMax 1.0)</tt>\n
  *   Default/recommended value: 1.0. This parameter has no influence when AutomaticParameterEstimation
  *   is used. In that case, always a value 1.0 is used.
  * \parameter SigmoidMin: The minimum of the sigmoid function (\f$f_{min}\f$). Must be smaller than 0.
  *   The parameter can be specified for each resolution, or for all resolutions at once.\n
  *   example: <tt>(SigmoidMin -0.8)</tt>\n
  *   Default value: -0.8. This parameter has no influence when AutomaticParameterEstimation
  *   is used. In that case, the value is automatically determined, depending on the images,
  *   metric etc.
  * \parameter SigmoidScale: The scale/width of the sigmoid function (\f$\omega\f$).
  *   The parameter can be specified for each resolution, or for all resolutions at once.\n
  *   example: <tt>(SigmoidScale 0.00001)</tt>\n
  *   Default value: 1e-8. This parameter has no influence when AutomaticParameterEstimation
  *   is used. In that case, the value is automatically determined, depending on the images,
  *   metric etc.
  * \parameter SigmoidInitialTime: the initial time input for the sigmoid (\f$t_0\f$). Must be
  *   larger than 0.0.
  *   The parameter can be specified for each resolution, or for all resolutions at once.\n
  *   example: <tt>(SigmoidInitialTime 0.0 5.0 5.0)</tt>\n
  *   Default value: 0.0. When increased, the optimization starts with smaller steps, leaving
  *   the possibility to increase the steps when necessary. If set to 0.0, the method starts with
  *   with the largest step allowed.
  * \parameter NumberOfGradientMeasurements: Number of gradients N to estimate the
  *   average square magnitudes of the exact gradient and the approximation error.
  *   The parameter can be specified for each resolution, or for all resolutions at once.\n
  *   example: <tt>(NumberOfGradientMeasurements 10)</tt>\n
  *   Default value: 0, which means that the value is automatically estimated.
  *   In principle, the more the better, but the slower. In practice N=10 is usually sufficient.
  *   But the automatic estimation achieved by N=0 also works good.
  *   The parameter has only influence when AutomaticParameterEstimation is used.
  * \parameter NumberOfJacobianMeasurements: The number of voxels M where the Jacobian is measured,
  *   which is used to estimate the covariance matrix.
  *   The parameter can be specified for each resolution, or for all resolutions at once.\n
  *   example: <tt>(NumberOfJacobianMeasurements 5000 10000 20000)</tt>\n
  *   Default value: M = max( 1000, nrofparams ), with nrofparams the
  *   number of transform parameters. This is a rather crude rule of thumb,
  *   which seems to work in practice. In principle, the more the better, but the slower.
  *   The parameter has only influence when AutomaticParameterEstimation is used.
  * \parameter NumberOfSamplesForExactGradient: The number of image samples used to compute
  *   the 'exact' gradient. The samples are chosen on a uniform grid.
  *   The parameter can be specified for each resolution, or for all resolutions at once.\n
  *   example: <tt>(NumberOfSamplesForExactGradient 100000)</tt>\n
  *   Default/recommended: 100000. This works in general. If the image is smaller, the number
  *   of samples is automatically reduced. In principle, the more the better, but the slower.
  *   The parameter has only influence when AutomaticParameterEstimation is used.
  * \parameter ASGDParameterEstimationMethod: The ASGD parameter estimation method used
  *   in this optimizer.
  *   The parameter can be specified for each resolution.\n
  *   example: <tt>(ASGDParameterEstimationMethod "Original")</tt>\n
  *         or <tt>(ASGDParameterEstimationMethod "DisplacementDistribution")</tt>\n
  *   Default: Original.
  * \parameter MaximumDisplacementEstimationMethod: The suitable position selection method used only for
  *   displacement distribution estimation method.
  *   The parameter can be specified for each resolution.\n
  *   example: <tt>(MaximumDisplacementEstimationMethod "2sigma")</tt>\n
  *         or <tt>(MaximumDisplacementEstimationMethod "95percentile")</tt>\n
  *   Default: 2sigma.
  * \parameter NoiseCompensation: Selects whether or not to use noise compensation.
  *   The parameter can be specified for each resolution, or for all resolutions at once.\n
  *   example: <tt>(NoiseCompensation "true")</tt>\n
  *   Default/recommended: true.
  *
  * \todo: this class contains a lot of functional code, which actually does not belong here.
  *
  * \sa AdaptiveStochasticVarianceReducedGradientOptimizer
  * \ingroup Optimizers
  */

template <class TElastix>
class AdaptiveStochasticLBFGS :
  public itk::AdaptiveStochasticLBFGSOptimizer,
  public OptimizerBase<TElastix>
{
public:

  /** Standard ITK. */
  typedef AdaptiveStochasticLBFGS           Self;
  typedef AdaptiveStochasticLBFGSOptimizer  Superclass1;
  typedef OptimizerBase<TElastix>                     Superclass2;
  typedef itk::SmartPointer<Self>                     Pointer;
  typedef itk::SmartPointer<const Self>               ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( AdaptiveStochasticLBFGS,
    AdaptiveStochasticLBFGSOptimizer );

  /** Name of this class.
   * Use this name in the parameter file to select this specific optimizer.
   * example: <tt>(Optimizer "AdaptiveStochasticLBFGS")</tt>\n
   */
  elxClassNameMacro( "AdaptiveStochasticLBFGS" );

  /** Typedef's inherited from Superclass1. */
  typedef Superclass1::CostFunctionType               CostFunctionType;
  typedef Superclass1::CostFunctionPointer            CostFunctionPointer;
  typedef Superclass1::StopConditionType              StopConditionType;

  /** Typedef's inherited from Superclass2. */
  typedef typename Superclass2::ElastixType           ElastixType;
  typedef typename Superclass2::ElastixPointer        ElastixPointer;
  typedef typename Superclass2::ConfigurationType     ConfigurationType;
  typedef typename Superclass2::ConfigurationPointer  ConfigurationPointer;
  typedef typename Superclass2::RegistrationType      RegistrationType;
  typedef typename Superclass2::RegistrationPointer   RegistrationPointer;
  typedef typename Superclass2::ITKBaseType           ITKBaseType;
  typedef itk::SizeValueType                          SizeValueType;

//  typedef LineSearchOptimizer           LineSearchOptimizerType;

//  typedef LineSearchOptimizerType::Pointer LineSearchOptimizerPointer;

  /** Typedef for the ParametersType. */
  typedef typename Superclass1::ParametersType        ParametersType;

  typedef itk::LineSearchOptimizer                    LineSearchOptimizerType;

  typedef LineSearchOptimizerType::Pointer            LineSearchOptimizerPointer;
  typedef itk::MoreThuenteLineSearchOptimizer         LineOptimizerType;
  typedef LineOptimizerType::Pointer                  LineOptimizerPointer;

  /** Methods invoked by elastix, in which parameters can be set and
   * progress information can be printed.
   */
  virtual void BeforeRegistration( void );
  virtual void BeforeEachResolution( void );
  virtual void AfterEachResolution( void );
  virtual void AfterEachIteration( void );
  virtual void AfterRegistration( void );

  /** Check if any scales are set, and set the UseScales flag on or off;
   * after that call the superclass' implementation.
   */
  virtual void StartOptimization( void );

  /** LBFGS Update step. */
  virtual void LBFGSUpdate( void );

  /** AdvanceOneStep. */
  virtual void AdvanceOneStep( void );

  /** If automatic gain estimation is desired, then estimate SP_a, SP_alpha
   * SigmoidScale, SigmoidMax, SigmoidMin.
   * After that call Superclass' implementation.
   */
  virtual void ResumeOptimization( void );

  /** Stop optimization and pass on exception. */
  virtual void MetricErrorResponse( itk::ExceptionObject & err );

  /** Codes of stopping conditions
   * The MinimumStepSize stopcondition never occurs, but may
   * be implemented in inheriting classes *
  typedef enum {
    MaximumNumberOfIterations,
    MetricError,
    MinimumStepSize } StopConditionType;

  /** Stop optimization.
  * \sa StopOptimization */
  virtual void StopOptimization( void );

  /** Set/Get whether automatic parameter estimation is desired.
   * If true, make sure to set the maximum step length.
   *
   * The following parameters are automatically determined:
   * SP_a, SP_alpha (=1), SigmoidMin, SigmoidMax (=1),
   * SigmoidScale.
   * A usually suitable value for SP_A is 20, which is the
   * default setting, if not specified by the user.
   */
  itkSetMacro( AutomaticParameterEstimation, bool );
  itkGetConstMacro( AutomaticParameterEstimation, bool );

  /** Set/Get whether automatic LBFGS step size estimation is desired. */
  itkSetMacro( AutomaticLBFGSStepsizeEstimation, bool );
  itkGetConstMacro( AutomaticLBFGSStepsizeEstimation, bool );

  /** Set/Get maximum step length. */
  itkSetMacro( MaximumStepLength, double );
  itkGetConstMacro( MaximumStepLength, double );

  /** Set the MaximumNumberOfSamplingAttempts. */
  itkSetMacro( MaximumNumberOfSamplingAttempts, SizeValueType );

  /** Get the MaximumNumberOfSamplingAttempts. */
  itkGetConstReferenceMacro( MaximumNumberOfSamplingAttempts, SizeValueType );

//   /** Set the learning rate. */
//   itkSetMacro( LearningRate, double );
//
//   /** Get the learning rate. */
//   itkGetConstReferenceMacro( LearningRate, double);

  /** Set the number of iterations. */
  itkSetMacro( NumberOfIterations, unsigned long );

  /** Get the number of iterations. */
  itkGetConstReferenceMacro( NumberOfIterations, unsigned long );

  /** Get the current iteration number. */
  itkGetConstMacro( CurrentIteration, unsigned int );

  /** Get the inner LBFGSMemory. */
  itkGetConstMacro( LBFGSMemory, unsigned int );

  /** Get the current value. */
  itkGetConstReferenceMacro( Value, double );

  /** Get current gradient. */
  itkGetConstReferenceMacro( Gradient, DerivativeType );

  /** Get current search direction. */
  itkGetConstReferenceMacro( SearchDir, DerivativeType );

  /** Set the Previous Position. */
  itkSetMacro( PreviousPosition, ParametersType );

  /** Get the Previous Position. */
  itkGetConstReferenceMacro( PreviousPosition, ParametersType);

  /** Get the Previous gradient. */
  itkGetConstReferenceMacro( PreviousGradient, DerivativeType);

  /** Type to count and reference number of threads */
  typedef unsigned int  ThreadIdType;

  /** Set the number of threads. */
  void SetNumberOfThreads( ThreadIdType numberOfThreads )
  {
    this->m_Threader->SetNumberOfThreads( numberOfThreads );
  }
  //itkGetConstReferenceMacro( NumberOfThreads, ThreadIdType );
  itkSetMacro( UseMultiThread, bool );

protected:

  /** Protected typedefs */
  typedef typename RegistrationType::FixedImageType   FixedImageType;
  typedef typename RegistrationType::MovingImageType  MovingImageType;

  typedef typename FixedImageType::RegionType         FixedImageRegionType;
  typedef typename FixedImageType::IndexType          FixedImageIndexType;
  typedef typename FixedImageType::PointType          FixedImagePointType;
  typedef typename RegistrationType::ITKBaseType      itkRegistrationType;
  typedef typename itkRegistrationType::TransformType TransformType;
  typedef typename TransformType::JacobianType        JacobianType;
  typedef itk::ComputeJacobianTerms<
    FixedImageType,TransformType >                    ComputeJacobianTermsType;
  typedef typename JacobianType::ValueType            JacobianValueType;
  struct SettingsType { double a, A, alpha, fmax, fmin, omega; };
  typedef typename std::vector<SettingsType>          SettingsVectorType;

  typedef itk::ComputeDisplacementDistribution<
    FixedImageType,TransformType >                    ComputeDisplacementDistributionType;

  /** Samplers: */
  typedef itk::ImageSamplerBase<FixedImageType>       ImageSamplerBaseType;
  typedef typename ImageSamplerBaseType::Pointer      ImageSamplerBasePointer;
  typedef itk::ImageRandomSamplerBase<FixedImageType> ImageRandomSamplerBaseType;
  typedef typename
    ImageRandomSamplerBaseType::Pointer               ImageRandomSamplerBasePointer;
  typedef
    itk::ImageRandomCoordinateSampler<FixedImageType> ImageRandomCoordinateSamplerType;
  typedef typename
    ImageRandomCoordinateSamplerType::Pointer         ImageRandomCoordinateSamplerPointer;
  typedef typename ImageSamplerBaseType::ImageSampleType       ImageSampleType;

  /** Image random sampler. */
  typedef itk::ImageRandomSampler< FixedImageType >   ImageRandomSamplerType;
  typedef typename ImageRandomSamplerType::Pointer    ImageRandomSamplerPointer;
  typedef typename
    ImageRandomSamplerType::ImageSampleContainerType  ImageRadomSampleContainerType;
  typedef typename
    ImageRadomSampleContainerType::Pointer            ImageRadomSampleContainerPointer;

  /** Image grid sampler. */
  typedef itk::ImageGridSampler< FixedImageType >     ImageGridSamplerType;
  typedef typename ImageGridSamplerType::Pointer      ImageGridSamplerPointer;
  typedef typename
    ImageGridSamplerType::ImageSampleContainerType    ImageSampleContainerType;
  typedef typename ImageSampleContainerType::Pointer  ImageSampleContainerPointer;

  /** Other protected typedefs */
  typedef itk::Statistics::MersenneTwisterRandomVariateGenerator RandomGeneratorType;
  typedef ProgressCommand                             ProgressCommandType;
  typedef typename ProgressCommand::Pointer           ProgressCommandPointer;

  /** Typedefs for support of sparse Jacobians and AdvancedTransforms. */
  typedef JacobianType                                TransformJacobianType;
  itkStaticConstMacro( FixedImageDimension, unsigned int, FixedImageType::ImageDimension );
  itkStaticConstMacro( MovingImageDimension, unsigned int, MovingImageType::ImageDimension );
  typedef typename TransformType::ScalarType          CoordinateRepresentationType;
  typedef itk::AdvancedTransform<
    CoordinateRepresentationType,
    itkGetStaticConstMacro(FixedImageDimension),
    itkGetStaticConstMacro(MovingImageDimension) >    AdvancedTransformType;
  typedef typename
    AdvancedTransformType::NonZeroJacobianIndicesType NonZeroJacobianIndicesType;

  /** For L-BFGS usage. */
  typedef itk::Array< double >               RhoType;
  typedef std::vector< ParametersType >      SType;
  typedef std::vector< DerivativeType >      YType;
  typedef itk::Array< double >               DiagonalMatrixType;

  AdaptiveStochasticLBFGS();
  virtual ~AdaptiveStochasticLBFGS() {};

  /** Variable to store the automatically determined settings for each resolution. */
  SettingsVectorType m_SettingsVector;

  /** Some options for automatic parameter estimation. */
  SizeValueType m_NumberOfGradientMeasurements;
  SizeValueType m_NumberOfJacobianMeasurements;
  SizeValueType m_NumberOfSamplesForExactGradient;

  /** The transform stored as AdvancedTransform */
  typename AdvancedTransformType::Pointer           m_AdvancedTransform;

  /** RandomGenerator for AddRandomPerturbation. */
  typename RandomGeneratorType::Pointer             m_RandomGenerator;

  double m_SigmoidScaleFactor;

  /** Check if the transform is an advanced transform. Called by Initialize. */
  virtual void CheckForAdvancedTransform( void );

  /** Print the contents of the settings vector to elxout. */
  virtual void PrintSettingsVector( const SettingsVectorType & settings ) const;

  /** Select different method to estimate some reasonable values for the parameters
   * SP_a, SP_alpha (=1), SigmoidMin, SigmoidMax (=1), and
   * SigmoidScale.
   */
  virtual void AutomaticParameterEstimation( void );

  /** Original estimation method to get the reasonable values for the parameters
   * SP_a, SP_alpha (=1), SigmoidMin, SigmoidMax (=1), and
   * SigmoidScale.
   */
  virtual void AutomaticParameterEstimationOriginal( void );

  /** Estimates some reasonable values for the parameters using displacement distribution
   * SP_a, SP_alpha (=1)
   */
  virtual void AutomaticParameterEstimationUsingDisplacementDistribution( void );

  virtual void AutomaticLBFGSStepsizeEstimation( void );

  /** Measure some derivatives, exact and approximated. Returns
   * the squared magnitude of the gradient and approximation error.
   * Needed for the automatic parameter estimation.
   * Gradients are measured at position mu_n, which are generated according to:
   * mu_n - mu_0 ~ N(0, perturbationSigma^2 I );
   * gg = g^T g, etc.
   */
  virtual void SampleGradients( const ParametersType & mu0,
    double perturbationSigma, double & gg, double & ee );

  /** Helper function, which calls GetScaledValueAndDerivative and does
   * some exception handling. Used by SampleGradients.
   */
  virtual void GetScaledDerivativeWithExceptionHandling(
    const ParametersType & parameters, DerivativeType & derivative );

  /** Helper function that adds a random perturbation delta to the input
   * parameters, with delta ~ sigma * N(0,I). Used by SampleGradients.
   */
  virtual void AddRandomPerturbation( ParametersType & parameters, double sigma );

  /** Store s = x_k - x_k-1 and y = g_k - g_k-1 in m_S and m_Y,
   * and store 1/(ys) in m_Rho. */
  virtual void StoreCurrentPoint(
    const ParametersType & step,
    const DerivativeType & grad_dif );
  /** Compute H0
   *
   * Override this method if not satisfied with the default choice.
   */
  virtual void ComputeDiagonalMatrix( DiagonalMatrixType & diag_H0 );

  /** Compute -Hg
   *
   *     COMPUTE -H*G USING THE FORMULA GIVEN IN: Nocedal, J. 1980,
   *     "Updating quasi-Newton matrices with limited storage",
   *     Mathematics of Computation, Vol.24, No.151, pp. 773-782.
   */
  virtual void ComputeSearchDirection(
    const DerivativeType & gradient,
    DerivativeType & searchDir );

  /** Setting: the minimum gradient magnitude.
   *
   * The optimizer stops when:
   * ||CurrentGradient|| < GradientMagnitudeTolerance * max(1, ||CurrentPosition||)
   */
  //itkGetConstMacro( GradientMagnitudeTolerance, double );
  //itkSetMacro( GradientMagnitudeTolerance, double );

  //double                        m_Value;
  //DerivativeType                m_Gradient;
  //double                        m_LearningRate;
  //StopConditionType             m_StopCondition;
  //DerivativeType                m_PreviousGradient;
  DerivativeType                m_PreviousCurvatureGradient;
  DerivativeType                m_SearchDir;
  ThreaderType::Pointer         m_Threader;

  //bool                          m_Stop;
  unsigned long                 m_NumberOfIterations;
  unsigned long                 m_CurrentIteration;
  double                        m_NoiseFactor;
  unsigned long                 m_LBFGSMemory;
  unsigned int                  m_CurrentT;
  unsigned int                  m_PreviousT;
  unsigned int                  m_Bound;

  RhoType m_Rho;
  SType   m_S;
  YType   m_Y;
  RhoType m_HessianFillValue;
  double  m_WindowScale;

private:

  AdaptiveStochasticLBFGS( const Self& );  // purposely not implemented
  void operator=( const Self& );                     // purposely not implemented

  // multi-threaded AdvanceOneStep:
  bool m_UseMultiThread;
  struct MultiThreaderParameterType
  {
    ParametersType *  t_NewPosition;
    Self *            t_Optimizer;
  };

  bool m_UseOpenMP;
  bool m_UseEigen;

  /** The callback function. */
  static ITK_THREAD_RETURN_TYPE AdvanceOneStepThreaderCallback( void * arg );

  /** The threaded implementation of AdvanceOneStep(). */
  inline void ThreadedAdvanceOneStep( ThreadIdType threadId, ParametersType & newPosition );

  bool    m_AutomaticParameterEstimation;
  bool    m_AutomaticLBFGSStepsizeEstimation;
  double  m_MaximumStepLength;

  /** Private variables for the sampling attempts. */
  SizeValueType m_MaximumNumberOfSamplingAttempts;
  SizeValueType m_CurrentNumberOfSamplingAttempts;
  SizeValueType m_PreviousErrorAtIteration;
  bool          m_AutomaticParameterEstimationDone;

  SizeValueType m_OutsideIterations;

  /** Private variables for band size estimation of covariance matrix. */
  SizeValueType m_MaxBandCovSize;
  SizeValueType m_NumberOfBandStructureSamples;
  SizeValueType m_NumberOfInnerLoopSamples;
  SizeValueType m_NumberOfSpatialSamples;

  /** The flag of using noise compensation. */
  bool m_UseNoiseCompensation;
  bool m_OriginalButSigmoidToDefault;
  bool m_UseNoiseFactor;

  bool                        m_UseAdaptiveLBFGSStepSizes;
  double                      m_GradientMagnitudeTolerance;

}; // end class AdaptiveStochasticLBFGS


} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxAdaptiveStochasticLBFGS.hxx"
#endif

#endif // end #ifndef __elxAdaptiveStochasticLBFGS_h
