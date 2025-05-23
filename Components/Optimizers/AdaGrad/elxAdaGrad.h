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
#ifndef elxAdaGrad_h
#define elxAdaGrad_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkAdaptiveStepsizeOptimizer.h"

#include "itkComputePreconditionerUsingDisplacementDistribution.h"
#include "itkComputeDisplacementDistribution.h" // For fast step size estimation

#include "elxProgressCommand.h"
#include "itkAdvancedTransform.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"
#include "itkAdvancedBSplineDeformableTransformBase.h"
#include "itkImageRandomSampler.h"


namespace elastix
{
/**
 * \class AdaGrad
 * \brief A gradient descent optimizer with an adaptive gain.
 *
 * This class is a wrap around the AdaptiveStepsizeOptimizer class.
 * It takes care of setting parameters and printing progress information.
 * For more information about the optimization method, please read the documentation
 * of the AdaptiveStepsizeOptimizer class.
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
 * "Adaptive stochastic gradient descent optimization for image registration,"
 * International Journal of Computer Vision, vol. 81, no. 3, pp. 227-239, 2009.
 * http://dx.doi.org/10.1007/s11263-008-0168-y
 *
 * Acceleration in case of many transform parameters was proposed in the following paper:
 *
 * [3] Y. Qiao, B. van Lew, B.P.F. Lelieveldt and M. Staring
 * "Fast Automatic Step Size Estimation for Gradient Descent Optimization of Image Registration,"
 * IEEE Transactions on Medical Imaging, vol. 35, no. 2, pp. 391 - 403, February 2016.
 * http://elastix.dev/marius/publications/2016_j_TMIa.php
 *
 * The parameters used in this class are:
 * \parameter Optimizer: Select this optimizer as follows:\n
 *   <tt>(Optimizer "AdaGrad")</tt>
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
 * \parameter StepSizeStrategy: When this parameter is set to "true", the adaptive
 *   step size mechanism described in the documentation of
 *   itk::itkAdaptiveStepsizeOptimizer is used.
 *   The parameter can be specified for each resolution, or for all resolutions at once.\n
 *   example: <tt>(StepSizeStrategy "Adaptive")</tt>\n
 *   Default/recommend value: "Adaptive", because it makes the registration more robust. In case
 *   of using a RandomCoordinate sampler, with (UseRandomSampleRegion "true"), the adaptive
 *   step size mechanism is turned off, no matter the user setting.
 * \parameter MaximumStepLength: Also called \f$\delta\f$. This parameter can be considered as
 *   the maximum voxel displacement between two iterations. The larger this parameter, the
 *   more aggressive the optimization.
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
 * \parameter NumberOfSamplesForNoiseCompensationFactor: The number of image samples used to compute
 *   the 'exact' gradient. The samples are chosen on a uniform grid.
 *   The parameter can be specified for each resolution, or for all resolutions at once.\n
 *   example: <tt>(NumberOfSamplesForNoiseCompensationFactor 100000)</tt>\n
 *   Default/recommended: 100000. This works in general. If the image is smaller, the number
 *   of samples is automatically reduced. In principle, the more the better, but the slower.
 *   The parameter has only influence when AutomaticParameterEstimation is used.
 * \parameter m_NumberOfSamplesForPrecondition: The number of image samples used to compute
 *   the gradient for preconditioner. The samples are chosen on a random sampler.
 *   The parameter can be specified for each resolution, or for all resolutions at once.\n
 *   example: <tt>(NumberOfSamplesForPrecondition 500000)</tt>\n
 *   Default/recommended: 500000. This works in general. If the image is smaller, the number
 *   of samples is automatically reduced. In principle, the more the better, but the slower.
 *   The parameter has only influence when AutomaticParameterEstimation is used.
 * \parameter RegularizationKappa: Selects for the preconditioner regularization.
 *   The parameter can be specified for each resolution, or for all resolutions at once.\n
 *   example: <tt>(RegularizationKappa 0.9)</tt>\n
 *
 * \todo: this class contains a lot of functional code, which actually does not belong here.
 *
 * \sa AdaptiveStepsizeOptimizer
 * \ingroup Optimizers
 */

template <typename TElastix>
class ITK_TEMPLATE_EXPORT AdaGrad
  : public itk::AdaptiveStepsizeOptimizer
  , public OptimizerBase<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(AdaGrad);

  /** Standard ITK. */
  using Self = AdaGrad;
  using Superclass1 = AdaptiveStepsizeOptimizer;
  using Superclass2 = OptimizerBase<TElastix>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(AdaGrad);

  /** Name of this class.
   * Use this name in the parameter file to select this specific optimizer.
   * example: <tt>(Optimizer "AdaGrad")</tt>\n
   */
  elxClassNameMacro("AdaGrad");

  /** Typedef's inherited from Superclass1. */
  using Superclass1::CostFunctionType;
  using Superclass1::CostFunctionPointer;
  using Superclass1::StopConditionType;

  /** Typedef's inherited from Superclass2. */
  using typename Superclass2::ElastixType;
  using typename Superclass2::RegistrationType;
  using ITKBaseType = typename Superclass2::ITKBaseType;
  using SizeValueType = itk::SizeValueType;

  /** Typedef for the ParametersType. */
  using typename Superclass1::ParametersType;

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

  /** Advance one step following the gradient direction. */
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

  /** Set/Get maximum step length. */
  itkSetMacro(MaximumStepLength, double);
  itkGetConstReferenceMacro(MaximumStepLength, double);

  /** Set/Get regularization value kappa. */
  itkSetClampMacro(RegularizationKappa, double, 0.0, 1.0);
  itkGetConstReferenceMacro(RegularizationKappa, double);

  /** Set/Get the MaximumNumberOfSamplingAttempts. */
  itkSetMacro(MaximumNumberOfSamplingAttempts, SizeValueType);
  itkGetConstReferenceMacro(MaximumNumberOfSamplingAttempts, SizeValueType);

protected:
  AdaGrad();
  ~AdaGrad() override = default;

  /** Protected typedefs */
  using FixedImageType = typename RegistrationType::FixedImageType;
  using MovingImageType = typename RegistrationType::MovingImageType;

  using FixedImageRegionType = typename FixedImageType::RegionType;
  using FixedImageIndexType = typename FixedImageType::IndexType;
  using FixedImagePointType = typename FixedImageType::PointType;
  using itkRegistrationType = typename RegistrationType::ITKBaseType;
  using TransformType = typename itkRegistrationType::TransformType;
  using JacobianType = typename TransformType::JacobianType;
  using JacobianValueType = typename JacobianType::ValueType;
  using typename Superclass2::SettingsType;
  using typename Superclass2::SettingsVectorType;
  using OutputImageType = typename ElastixType::FixedImageType;

  using PreconditionerEstimationType =
    itk::ComputePreconditionerUsingDisplacementDistribution<FixedImageType, TransformType>;
  using PreconditionerEstimationPointer = typename PreconditionerEstimationType::Pointer;

  using ComputeDisplacementDistributionType = itk::ComputeDisplacementDistribution<FixedImageType, TransformType>;


  /** Samplers: */
  using ImageSamplerBaseType = itk::ImageSamplerBase<FixedImageType>;
  using ImageSamplerBasePointer = typename ImageSamplerBaseType::Pointer;
  using ImageRandomSamplerBaseType = itk::ImageRandomSamplerBase<FixedImageType>;
  using ImageRandomSamplerBasePointer = typename ImageRandomSamplerBaseType::Pointer;
  using ImageRandomCoordinateSamplerType = itk::ImageRandomCoordinateSampler<FixedImageType>;
  using ImageRandomCoordinateSamplerPointer = typename ImageRandomCoordinateSamplerType::Pointer;
  using ImageRandomSamplerType = itk::ImageRandomSampler<FixedImageType>;
  using ImageRandomSamplerPointer = typename ImageRandomSamplerType::Pointer;
  using ImageGridSamplerType = itk::ImageGridSampler<FixedImageType>;
  using ImageGridSamplerPointer = typename ImageGridSamplerType::Pointer;
  using ImageSampleContainerType = typename ImageGridSamplerType::ImageSampleContainerType;
  using ImageSampleContainerPointer = typename ImageSampleContainerType::Pointer;

  /** Other protected typedefs */
  using RandomGeneratorType = itk::Statistics::MersenneTwisterRandomVariateGenerator;
  using RandomGeneratorPointer = typename RandomGeneratorType::Pointer;

  /** Typedefs for support of sparse Jacobians and AdvancedTransforms. */
  using TransformJacobianType = JacobianType;
  itkStaticConstMacro(FixedImageDimension, unsigned int, FixedImageType::ImageDimension);
  itkStaticConstMacro(MovingImageDimension, unsigned int, MovingImageType::ImageDimension);
  using CoordinateRepresentationType = typename TransformType::ScalarType;

  using AdvancedTransformType =
    itk::AdvancedTransform<CoordinateRepresentationType, Self::FixedImageDimension, Self::MovingImageDimension>;
  using AdvancedTransformPointer = typename AdvancedTransformType::Pointer;
  using NonZeroJacobianIndicesType = typename AdvancedTransformType::NonZeroJacobianIndicesType;

  using AdvancedBSplineDeformableTransformType =
    itk::AdvancedBSplineDeformableTransformBase<CoordinateRepresentationType, Self::FixedImageDimension>;
  using BSplineTransformBasePointer = typename AdvancedBSplineDeformableTransformType::Pointer;

  /** Variable to store the automatically determined settings for each resolution. */
  SettingsVectorType m_SettingsVector;

  /** Some options for automatic parameter estimation. */
  SizeValueType m_NumberOfGradientMeasurements;
  SizeValueType m_NumberOfJacobianMeasurements;
  SizeValueType m_NumberOfSamplesForNoiseCompensationFactor;
  SizeValueType m_NumberOfSamplesForPrecondition;
  SizeValueType m_NumberOfSpatialSamples;

  /** The transform stored as AdvancedTransform */
  AdvancedTransformPointer m_AdvancedTransform;

  double m_SigmoidScaleFactor;
  double m_NoiseFactor;
  double m_GlobalStepSize;
  double m_RegularizationKappa;
  double m_ConditionNumber;

  /** Select different method to estimate some reasonable values for the parameters
   * SP_a, SP_alpha (=1), SigmoidMin, SigmoidMax (=1), and
   * SigmoidScale.
   */
  virtual void
  AutomaticPreconditionerEstimation();

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

private:
  elxOverrideGetSelfMacro;

  bool   m_AutomaticParameterEstimation;
  double m_MaximumStepLength;
  double m_MaximumStepLengthRatio;

  /** Private variables for the sampling attempts. */
  SizeValueType m_MaximumNumberOfSamplingAttempts;
  SizeValueType m_CurrentNumberOfSamplingAttempts;
  SizeValueType m_PreviousErrorAtIteration;
  bool          m_AutomaticParameterEstimationDone;

  /** Private variables for band size estimation of covariance matrix. */
  SizeValueType m_MaxBandCovSize;
  SizeValueType m_NumberOfBandStructureSamples;

  /** The flag of using noise compensation. */
  bool m_UseNoiseCompensation;
  bool m_OriginalButSigmoidToDefault;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxAdaGrad.hxx"
#endif

#endif // end #ifndef elxAdaGrad_h
