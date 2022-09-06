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
#ifndef elxAdaptiveStochasticLBFGS_h
#define elxAdaptiveStochasticLBFGS_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkAdaptiveStochasticLBFGSOptimizer.h"

#include "elxProgressCommand.h"
#include "itkAdvancedTransform.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"
#include "itkComputeJacobianTerms.h"
#include "itkComputeDisplacementDistribution.h"
#include "itkPlatformMultiThreader.h"
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
 * [1]  Y.Qiao, Z.Sun, B.P.F. Lelieveldt, M.Staring
 * A stochastic quasi-newton method for non-rigid image registration
 * Medical Image Computing and Computer-Assisted Intervention (MICCAI), pp. 297-304, 2015.
 * http://dx.doi.org/10.1007/978-3-319-24571-3_36
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
 *   more aggressive the optimization.
 *   The parameter can be specified for each resolution, or for all resolutions at once.\n
 *   example: <tt>(MaximumStepLength 1.0)</tt>\n
 *   Default: mean voxel spacing of fixed and moving image. This seems to work well in general.
 *   This parameter only has influence when AutomaticParameterEstimation is used.
 *
 * \todo: this class contains a lot of functional code, which actually does not belong here.
 *
 * \sa AdaptiveStochasticLBFGS
 * \ingroup Optimizers
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT AdaptiveStochasticLBFGS
  : public itk::AdaptiveStochasticLBFGSOptimizer
  , public OptimizerBase<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(AdaptiveStochasticLBFGS);

  /** Standard ITK. */
  using Self = AdaptiveStochasticLBFGS;
  using Superclass1 = AdaptiveStochasticLBFGSOptimizer;
  using Superclass2 = OptimizerBase<TElastix>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(AdaptiveStochasticLBFGS, AdaptiveStochasticLBFGSOptimizer);

  /** Name of this class.
   * Use this name in the parameter file to select this specific optimizer.
   * example: <tt>(Optimizer "AdaptiveStochasticLBFGS")</tt>\n
   */
  elxClassNameMacro("AdaptiveStochasticLBFGS");

  /** Typedef's inherited from Superclass1. */
  using Superclass1::CostFunctionType;
  using Superclass1::CostFunctionPointer;
  using Superclass1::StopConditionType;

  /** Typedef's inherited from Superclass2. */
  using typename Superclass2::ElastixType;
  using typename Superclass2::RegistrationType;
  using ITKBaseType = typename Superclass2::ITKBaseType;
  using SizeValueType = itk::SizeValueType;

  //  using LineSearchOptimizerType = LineSearchOptimizer          ;

  //  using LineSearchOptimizerPointer = LineSearchOptimizerType::Pointer;

  /** Typedef for the ParametersType. */
  using typename Superclass1::ParametersType;

  using LineSearchOptimizerType = itk::LineSearchOptimizer;

  using LineSearchOptimizerPointer = LineSearchOptimizerType::Pointer;
  using LineOptimizerType = itk::MoreThuenteLineSearchOptimizer;
  using LineOptimizerPointer = LineOptimizerType::Pointer;

  /** Methods invoked by elastix, in which parameters can be set and
   * progress information can be printed.
   */
  void
  BeforeRegistration() override;
  void
  BeforeEachResolution() override;
  void
  AfterEachResolution() override;
  void
  AfterEachIteration() override;
  void
  AfterRegistration() override;

  /** Check if any scales are set, and set the UseScales flag on or off;
   * after that call the superclass' implementation.
   */
  void
  StartOptimization() override;

  /** LBFGS Update step. */
  virtual void
  LBFGSUpdate();

  /** AdvanceOneStep. */
  void
  AdvanceOneStep() override;

  /** If automatic gain estimation is desired, then estimate SP_a, SP_alpha
   * SigmoidScale, SigmoidMax, SigmoidMin.
   * After that call Superclass' implementation.
   */
  void
  ResumeOptimization() override;

  /** Stop optimization and pass on exception. */
  void
  MetricErrorResponse(itk::ExceptionObject & err) override;

  /** Stop optimization.
   * \sa StopOptimization */
  void
  StopOptimization() override;

  /** Set/Get whether automatic parameter estimation is desired.
   * If true, make sure to set the maximum step length.
   *
   * The following parameters are automatically determined:
   * SP_a, SP_alpha (=1), SigmoidMin, SigmoidMax (=1),
   * SigmoidScale.
   * A usually suitable value for SP_A is 20, which is the
   * default setting, if not specified by the user.
   */
  itkSetMacro(AutomaticParameterEstimation, bool);
  itkGetConstMacro(AutomaticParameterEstimation, bool);

  /** Set/Get whether automatic LBFGS step size estimation is desired. */
  itkSetMacro(AutomaticLBFGSStepsizeEstimation, bool);
  itkGetConstMacro(AutomaticLBFGSStepsizeEstimation, bool);

  /** Set/Get maximum step length. */
  itkSetMacro(MaximumStepLength, double);
  itkGetConstMacro(MaximumStepLength, double);

  /** Set the MaximumNumberOfSamplingAttempts. */
  itkSetMacro(MaximumNumberOfSamplingAttempts, SizeValueType);

  /** Get the MaximumNumberOfSamplingAttempts. */
  itkGetConstReferenceMacro(MaximumNumberOfSamplingAttempts, SizeValueType);

  /** Type to count and reference number of threads */
  using ThreadIdType = unsigned int;

  /** Set the number of threads. */
  void
  SetNumberOfWorkUnits(ThreadIdType numberOfThreads)
  {
    this->m_Threader->SetNumberOfWorkUnits(numberOfThreads);
  }

protected:
  /** Protected typedefs */
  using FixedImageType = typename RegistrationType::FixedImageType;
  using MovingImageType = typename RegistrationType::MovingImageType;

  using FixedImageRegionType = typename FixedImageType::RegionType;
  using FixedImageIndexType = typename FixedImageType::IndexType;
  using FixedImagePointType = typename FixedImageType::PointType;
  using itkRegistrationType = typename RegistrationType::ITKBaseType;
  using TransformType = typename itkRegistrationType::TransformType;
  using JacobianType = typename TransformType::JacobianType;
  using ComputeJacobianTermsType = itk::ComputeJacobianTerms<FixedImageType, TransformType>;
  using JacobianValueType = typename JacobianType::ValueType;
  struct SettingsType
  {
    double a, A, alpha, fmax, fmin, omega;
  };
  using SettingsVectorType = typename std::vector<SettingsType>;

  using ComputeDisplacementDistributionType = itk::ComputeDisplacementDistribution<FixedImageType, TransformType>;

  /** Samplers: */
  using ImageSamplerBaseType = itk::ImageSamplerBase<FixedImageType>;
  using ImageSamplerBasePointer = typename ImageSamplerBaseType::Pointer;
  using ImageRandomSamplerBaseType = itk::ImageRandomSamplerBase<FixedImageType>;
  using ImageRandomSamplerBasePointer = typename ImageRandomSamplerBaseType::Pointer;
  using ImageRandomCoordinateSamplerType = itk::ImageRandomCoordinateSampler<FixedImageType>;
  using ImageRandomCoordinateSamplerPointer = typename ImageRandomCoordinateSamplerType::Pointer;
  using ImageSampleType = typename ImageSamplerBaseType::ImageSampleType;

  /** Image random sampler. */
  using ImageRandomSamplerType = itk::ImageRandomSampler<FixedImageType>;
  using ImageRandomSamplerPointer = typename ImageRandomSamplerType::Pointer;
  using ImageRadomSampleContainerType = typename ImageRandomSamplerType::ImageSampleContainerType;
  using ImageRadomSampleContainerPointer = typename ImageRadomSampleContainerType::Pointer;

  /** Image grid sampler. */
  using ImageGridSamplerType = itk::ImageGridSampler<FixedImageType>;
  using ImageGridSamplerPointer = typename ImageGridSamplerType::Pointer;
  using ImageSampleContainerType = typename ImageGridSamplerType::ImageSampleContainerType;
  using ImageSampleContainerPointer = typename ImageSampleContainerType::Pointer;

  /** Other protected typedefs */
  using RandomGeneratorType = itk::Statistics::MersenneTwisterRandomVariateGenerator;
  using ProgressCommandType = ProgressCommand;
  using ProgressCommandPointer = typename ProgressCommand::Pointer;

  /** Typedefs for support of sparse Jacobians and AdvancedTransforms. */
  using TransformJacobianType = JacobianType;
  itkStaticConstMacro(FixedImageDimension, unsigned int, FixedImageType::ImageDimension);
  itkStaticConstMacro(MovingImageDimension, unsigned int, MovingImageType::ImageDimension);
  using CoordinateRepresentationType = typename TransformType::ScalarType;
  using AdvancedTransformType =
    itk::AdvancedTransform<CoordinateRepresentationType, Self::FixedImageDimension, Self::MovingImageDimension>;
  using NonZeroJacobianIndicesType = typename AdvancedTransformType::NonZeroJacobianIndicesType;

  /** For L-BFGS usage. */
  using RhoType = itk::Array<double>;
  using SType = std::vector<ParametersType>;
  using YType = std::vector<DerivativeType>;
  using DiagonalMatrixType = itk::Array<double>;

  AdaptiveStochasticLBFGS();
  ~AdaptiveStochasticLBFGS() override = default;

  /** Variable to store the automatically determined settings for each resolution. */
  SettingsVectorType m_SettingsVector;

  /** Some options for automatic parameter estimation. */
  SizeValueType m_NumberOfGradientMeasurements;
  SizeValueType m_NumberOfJacobianMeasurements;
  SizeValueType m_NumberOfSamplesForExactGradient;

  /** The transform stored as AdvancedTransform */
  typename AdvancedTransformType::Pointer m_AdvancedTransform;

  /** RandomGenerator for AddRandomPerturbation. */
  typename RandomGeneratorType::Pointer m_RandomGenerator;

  double m_SigmoidScaleFactor;

  /** Print the contents of the settings vector to elxout. */
  virtual void
  PrintSettingsVector(const SettingsVectorType & settings) const;

  /** Select different method to estimate some reasonable values for the parameters
   * SP_a, SP_alpha (=1), SigmoidMin, SigmoidMax (=1), and
   * SigmoidScale.
   */
  virtual void
  AutomaticParameterEstimation();

  /** Original estimation method to get the reasonable values for the parameters
   * SP_a, SP_alpha (=1), SigmoidMin, SigmoidMax (=1), and
   * SigmoidScale.
   */
  virtual void
  AutomaticParameterEstimationOriginal();

  /** Estimates some reasonable values for the parameters using displacement distribution
   * SP_a, SP_alpha (=1)
   */
  virtual void
  AutomaticParameterEstimationUsingDisplacementDistribution();

  virtual void
  AutomaticLBFGSStepsizeEstimation();

  /** Measure some derivatives, exact and approximated. Returns
   * the squared magnitude of the gradient and approximation error.
   * Needed for the automatic parameter estimation.
   * Gradients are measured at position mu_n, which are generated according to:
   * mu_n - mu_0 ~ N(0, perturbationSigma^2 I );
   * gg = g^T g, etc.
   */
  virtual void
  SampleGradients(const ParametersType & mu0, double perturbationSigma, double & gg, double & ee);

  /** Helper function, which calls GetScaledValueAndDerivative and does
   * some exception handling. Used by SampleGradients.
   */
  virtual void
  GetScaledDerivativeWithExceptionHandling(const ParametersType & parameters, DerivativeType & derivative);

  /** Helper function that adds a random perturbation delta to the input
   * parameters, with delta ~ sigma * N(0,I). Used by SampleGradients.
   */
  virtual void
  AddRandomPerturbation(ParametersType & parameters, double sigma);

  /** Store s = x_k - x_k-1 and y = g_k - g_k-1 in m_S and m_Y,
   * and store 1/(ys) in m_Rho. */
  virtual void
  StoreCurrentPoint(const ParametersType & step, const DerivativeType & grad_dif);
  /** Compute H0
   *
   * Override this method if not satisfied with the default choice.
   */
  virtual void
  ComputeDiagonalMatrix(DiagonalMatrixType & diag_H0);

  /** Compute -Hg
   *
   *     COMPUTE -H*G USING THE FORMULA GIVEN IN: Nocedal, J. 1980,
   *     "Updating quasi-Newton matrices with limited storage",
   *     Mathematics of Computation, Vol.24, No.151, pp. 773-782.
   */
  virtual void
  ComputeSearchDirection(const DerivativeType & gradient, DerivativeType & searchDir);

  /** Setting: the minimum gradient magnitude.
   *
   * The optimizer stops when:
   * ||CurrentGradient|| < GradientMagnitudeTolerance * max(1, ||CurrentPosition||)
   */

  DerivativeType m_PreviousCurvatureGradient;

  double       m_NoiseFactor;
  unsigned int m_CurrentT;
  unsigned int m_PreviousT;
  unsigned int m_Bound;

  RhoType m_Rho;
  SType   m_S;
  YType   m_Y;
  RhoType m_HessianFillValue;
  double  m_WindowScale;

private:
  elxOverrideGetSelfMacro;

  // multi-threaded AdvanceOneStep:
  struct MultiThreaderParameterType
  {
    ParametersType * t_NewPosition;
    Self *           t_Optimizer;
  };

  /** The callback function. */
  static itk::ITK_THREAD_RETURN_TYPE
  AdvanceOneStepThreaderCallback(void * arg);

  /** The threaded implementation of AdvanceOneStep(). */
  inline void
  ThreadedAdvanceOneStep(ThreadIdType threadId, ParametersType & newPosition);

  bool   m_AutomaticParameterEstimation;
  bool   m_AutomaticLBFGSStepsizeEstimation;
  double m_MaximumStepLength;

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

  bool m_UseAdaptiveLBFGSStepSizes;

}; // end class AdaptiveStochasticLBFGS


} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxAdaptiveStochasticLBFGS.hxx"
#endif

#endif // end #ifndef elxAdaptiveStochasticLBFGS_h
