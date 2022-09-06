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
#ifndef elxPreconditionedGradientDescent_h
#define elxPreconditionedGradientDescent_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkAdaptiveStochasticPreconditionedGradientDescentOptimizer.h"

#include "itkImageGridSampler.h"
#include "itkImageRandomCoordinateSampler.h"
#include "elxProgressCommand.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"

namespace elastix
{
/**
 * \class PreconditionedGradientDescent
 * \brief A gradient descent optimizer with a decaying gain.
 *
 * This class is a wrap around the AdaptiveStochasticPreconditionedGradientOptimizer class.
 * It takes care of setting parameters and printing progress information.
 * For more information about the optimization method, please read the documentation
 * of the AdaptiveStochasticPreconditionedGradientOptimizer class.
 *
 * The parameters used in this class are:
 * \parameter Optimizer: Select this optimizer as follows:\n
 *   <tt>(Optimizer "PreconditionedGradientDescent")</tt>
 * \parameter MaximumNumberOfIterations: The maximum number of iterations in each resolution. \n
 *   example: <tt>(MaximumNumberOfIterations 100 100 50)</tt> \n
 *    Default/recommended value: 500.
 * \parameter MaximumNumberOfSamplingAttempts: The maximum number of sampling attempts. Sometimes
 *   not enough corresponding samples can be drawn, upon which an exception is thrown. With this
 *   parameter it is possible to try to draw another set of samples. \n
 *   example: <tt>(MaximumNumberOfSamplingAttempts 10 15 10)</tt> \n
 *    Default value: 0, i.e. just fail immediately, for backward compatibility.
 * \parameter SP_a: The gain \f$a(k)\f$ at each iteration \f$k\f$ is defined by \n
 *   \f$a(k) =  SP\_a / (SP\_A + k + 1)^{SP\_alpha}\f$. \n
 *   SP_a can be defined for each resolution. \n
 *   example: <tt>(SP_a 3200.0 3200.0 1600.0)</tt> \n
 *   The default value is 400.0. Tuning this variable for you specific problem is recommended.
 * \parameter SP_A: The gain \f$a(k)\f$ at each iteration \f$k\f$ is defined by \n
 *   \f$a(k) =  SP\_a / (SP\_A + k + 1)^{SP\_alpha}\f$. \n
 *   SP_A can be defined for each resolution. \n
 *   example: <tt>(SP_A 50.0 50.0 100.0)</tt> \n
 *   The default/recommended value is 50.0.
 * \parameter SP_alpha: The gain \f$a(k)\f$ at each iteration \f$k\f$ is defined by \n
 *   \f$a(k) =  SP\_a / (SP\_A + k + 1)^{SP\_alpha}\f$. \n
 *   SP_alpha can be defined for each resolution. \n
 *   example: <tt>(SP_alpha 0.602 0.602 0.602)</tt> \n
 *   The default/recommended value is 0.602.
 *
 * \sa StochasticPreconditionedGradientOptimizer
 * \ingroup Optimizers
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT PreconditionedGradientDescent
  : public itk::AdaptiveStochasticPreconditionedGradientDescentOptimizer
  , public OptimizerBase<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(PreconditionedGradientDescent);

  /** Standard ITK.*/
  using Self = PreconditionedGradientDescent;
  using Superclass1 = AdaptiveStochasticPreconditionedGradientDescentOptimizer;
  using Superclass2 = OptimizerBase<TElastix>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(PreconditionedGradientDescent, AdaptiveStochasticPreconditionedGradientDescentOptimizer);

  /** Name of this class.
   * Use this name in the parameter file to select this specific optimizer.
   * example: <tt>(Optimizer "PreconditionedGradientDescent")</tt>\n
   */
  elxClassNameMacro("PreconditionedGradientDescent");

  /** Typedef's inherited from Superclass1, the AdaptiveStochasticPreconditionedGradientDescentOptimizer. */
  using Superclass1::CostFunctionType;
  using Superclass1::CostFunctionPointer;
  using Superclass1::StopConditionType;

  /** Typedef's inherited from Superclass2, the elastix OptimizerBase. */
  using typename Superclass2::ElastixType;
  using typename Superclass2::RegistrationType;
  using ITKBaseType = typename Superclass2::ITKBaseType;

  /** Typedef for the ParametersType. */
  using typename Superclass1::ParametersType;

  /** Some typedefs for computing the SelfHessian */
  using typename Superclass1::PreconditionValueType;
  using typename Superclass1::PreconditionType;
  // using typename Superclass1::EigenSystemType;

  /** Methods invoked by elastix, in which parameters can be set and
   * progress information can be printed.
   */
  virtual void
  BeforeRegistration();
  virtual void
  BeforeEachResolution();
  virtual void
  AfterEachResolution();
  virtual void
  AfterEachIteration();
  virtual void
  AfterRegistration();

  /** Check if any scales are set, and set the UseScales flag on or off;
   * after that call the superclass' implementation.
   */
  virtual void
  StartOptimization();

  /** Stop optimization and pass on exception. */
  virtual void
  MetricErrorResponse(itk::ExceptionObject & err);

  /** Add SetCurrentPositionPublic, which calls the protected
   * SetCurrentPosition of the itkAdaptiveStochasticPreconditionedGradientDescentOptimizer class.
   */
  virtual void
  SetCurrentPositionPublic(const ParametersType & param)
  {
    this->Superclass1::SetCurrentPosition(param);
  }

  /** Set/Get whether automatic parameter estimation is desired.
   * If true, make sure to set the maximum step length.
   *
   * The following parameters are automatically determined:
   * SP_a = 2*SP_A, SP_alpha (=1), SigmoidMin, SigmoidMax (=1),
   * SigmoidScale.
   * A usually suitable value for SP_A is 20, which is the
   * default setting, if not specified by the user.
   */
  itkSetMacro(AutomaticParameterEstimation, bool);
  itkGetConstMacro(AutomaticParameterEstimation, bool);

  /** Set the MaximumNumberOfSamplingAttempts. */
  itkSetMacro(MaximumNumberOfSamplingAttempts, unsigned long);

  /** Get the MaximumNumberOfSamplingAttempts. */
  itkGetConstReferenceMacro(MaximumNumberOfSamplingAttempts, unsigned long);

  /** Set the SelfHessian as a preconditioning matrix and call Superclass' implementation.
   * Only done when m_PreconditionMatrixSet == false;
   * If automatic gain estimation is desired, then estimate SP_a, SP_alpha
   * SigmoidScale, SigmoidMax, SigmoidMin.
   * After that call Superclass' implementation.
   */
  virtual void
  ResumeOptimization();

protected:
  struct SettingsType
  {
    double a, A, alpha, fmax, fmin, omega;
  };
  using SettingsVectorType = typename std::vector<SettingsType>;

  /** Other protected typedefs */
  using RandomGeneratorType = itk::Statistics::MersenneTwisterRandomVariateGenerator;
  using RandomGeneratorPointer = typename RandomGeneratorType::Pointer;
  using ProgressCommandType = ProgressCommand;
  using ProgressCommandPointer = typename ProgressCommand::Pointer;

  /** Samplers: */
  using FixedImageType = typename RegistrationType::FixedImageType;
  using MovingImageType = typename RegistrationType::MovingImageType;
  using ImageSamplerBaseType = itk::ImageSamplerBase<FixedImageType>;
  using ImageSamplerBasePointer = typename ImageSamplerBaseType::Pointer;
  using ImageRandomSamplerBaseType = itk::ImageRandomSamplerBase<FixedImageType>;
  using ImageRandomSamplerBasePointer = typename ImageRandomSamplerBaseType::Pointer;
  using ImageRandomCoordinateSamplerType = itk::ImageRandomCoordinateSampler<FixedImageType>;
  using ImageRandomCoordinateSamplerPointer = typename ImageRandomCoordinateSamplerType::Pointer;
  using ImageGridSamplerType = itk::ImageGridSampler<FixedImageType>;
  using ImageGridSamplerPointer = typename ImageGridSamplerType::Pointer;

  PreconditionedGradientDescent();
  virtual ~PreconditionedGradientDescent(){};

  /** Variable to store the automatically determined settings for each resolution. */
  SettingsVectorType m_SettingsVector;

  unsigned int m_NumberOfGradientMeasurements;
  unsigned int m_NumberOfSamplesForExactGradient;
  double       m_SigmoidScaleFactor;

  /** RandomGenerator for AddRandomPerturbation. */
  RandomGeneratorPointer m_RandomGenerator;

  /** Get the SelfHessian from the metric and submit as Precondition matrix */
  virtual void
  SetSelfHessian();

  /** Print the contents of the settings vector to elxout. */
  virtual void
  PrintSettingsVector(const SettingsVectorType & settings) const;

  /** Estimates some reasonable values for the parameters
   * SP_a, SP_alpha (=1), SigmoidMin, SigmoidMax (=1), and SigmoidScale.
   */
  virtual void
  AutomaticParameterEstimation();

  /** Measure some derivatives, exact and approximated. Returns
   * the sigma1 and sigma2.
   * Needed for the automatic parameter estimation.
   * Gradients are measured at position mu_n, which are generated according to:
   * mu_n - mu_0 ~ N(0, perturbationSigma P );
   * perturbationSigma = sigma1 = 1/N g_0' P g_0,
   * where g_0 is the exact gradient at mu_0, and N the number of parameters.
   * sigma2 = 1/N sum_n g_n' P g_n / (sum_n n),
   * where g_n is the approximated gradient at mu_0.
   * NB: sigma1 = sqr(sigma_1), sigma2 = sqr(sigma_2), compared to my notes.
   */
  virtual void
  SampleGradients(const ParametersType & mu0, double & sigma1, double & sigma2);

  /** Helper function, which calls GetScaledValueAndDerivative and does
   * some exception handling. Used by SampleGradients.
   */
  virtual void
  GetScaledDerivativeWithExceptionHandling(const ParametersType & parameters, DerivativeType & derivative);

  /** Helper function that adds a random perturbation delta to the input
   * parameters, with delta ~ sigma * N(0,I). Used by SampleGradients.
   */
  virtual void
  AddRandomPerturbation(const ParametersType & initialParameters, ParametersType & perturbedParameters, double sigma);

private:
  elxOverrideGetSelfMacro;

  /** Private variables for the sampling attempts. */
  unsigned long m_MaximumNumberOfSamplingAttempts;
  unsigned long m_CurrentNumberOfSamplingAttempts;
  unsigned long m_PreviousErrorAtIteration;

  /** Private variables for self Hessian support. */
  bool m_PreconditionMatrixSet;

  bool m_AutomaticParameterEstimation;
  bool m_AutomaticParameterEstimationDone;

}; // end class PreconditionedGradientDescent


} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxPreconditionedGradientDescent.hxx"
#endif

#endif // end #ifndef elxPreconditionedGradientDescent_h
