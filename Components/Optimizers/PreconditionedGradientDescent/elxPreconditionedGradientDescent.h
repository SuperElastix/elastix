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
  /** Standard ITK.*/
  typedef PreconditionedGradientDescent                            Self;
  typedef AdaptiveStochasticPreconditionedGradientDescentOptimizer Superclass1;
  typedef OptimizerBase<TElastix>                                  Superclass2;
  typedef itk::SmartPointer<Self>                                  Pointer;
  typedef itk::SmartPointer<const Self>                            ConstPointer;

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
  typedef Superclass1::CostFunctionType    CostFunctionType;
  typedef Superclass1::CostFunctionPointer CostFunctionPointer;
  typedef Superclass1::StopConditionType   StopConditionType;

  /** Typedef's inherited from Superclass2, the elastix OptimizerBase. */
  typedef typename Superclass2::ElastixType          ElastixType;
  typedef typename Superclass2::ElastixPointer       ElastixPointer;
  typedef typename Superclass2::ConfigurationType    ConfigurationType;
  typedef typename Superclass2::ConfigurationPointer ConfigurationPointer;
  typedef typename Superclass2::RegistrationType     RegistrationType;
  typedef typename Superclass2::RegistrationPointer  RegistrationPointer;
  typedef typename Superclass2::ITKBaseType          ITKBaseType;

  /** Typedef for the ParametersType. */
  typedef typename Superclass1::ParametersType ParametersType;

  /** Some typedefs for computing the SelfHessian */
  typedef typename Superclass1::PreconditionValueType PreconditionValueType;
  typedef typename Superclass1::PreconditionType      PreconditionType;
  // typedef typename Superclass1::EigenSystemType           EigenSystemType;

  /** Methods invoked by elastix, in which parameters can be set and
   * progress information can be printed.
   */
  virtual void
  BeforeRegistration(void);
  virtual void
  BeforeEachResolution(void);
  virtual void
  AfterEachResolution(void);
  virtual void
  AfterEachIteration(void);
  virtual void
  AfterRegistration(void);

  /** Check if any scales are set, and set the UseScales flag on or off;
   * after that call the superclass' implementation.
   */
  virtual void
  StartOptimization(void);

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
  ResumeOptimization(void);

protected:
  struct SettingsType
  {
    double a, A, alpha, fmax, fmin, omega;
  };
  typedef typename std::vector<SettingsType> SettingsVectorType;

  /** Other protected typedefs */
  typedef itk::Statistics::MersenneTwisterRandomVariateGenerator RandomGeneratorType;
  typedef typename RandomGeneratorType::Pointer                  RandomGeneratorPointer;
  typedef ProgressCommand                                        ProgressCommandType;
  typedef typename ProgressCommand::Pointer                      ProgressCommandPointer;

  /** Samplers: */
  typedef typename RegistrationType::FixedImageType          FixedImageType;
  typedef typename RegistrationType::MovingImageType         MovingImageType;
  typedef itk::ImageSamplerBase<FixedImageType>              ImageSamplerBaseType;
  typedef typename ImageSamplerBaseType::Pointer             ImageSamplerBasePointer;
  typedef itk::ImageRandomSamplerBase<FixedImageType>        ImageRandomSamplerBaseType;
  typedef typename ImageRandomSamplerBaseType::Pointer       ImageRandomSamplerBasePointer;
  typedef itk::ImageRandomCoordinateSampler<FixedImageType>  ImageRandomCoordinateSamplerType;
  typedef typename ImageRandomCoordinateSamplerType::Pointer ImageRandomCoordinateSamplerPointer;
  typedef itk::ImageGridSampler<FixedImageType>              ImageGridSamplerType;
  typedef typename ImageGridSamplerType::Pointer             ImageGridSamplerPointer;

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
  SetSelfHessian(void);

  /** Print the contents of the settings vector to elxout. */
  virtual void
  PrintSettingsVector(const SettingsVectorType & settings) const;

  /** Estimates some reasonable values for the parameters
   * SP_a, SP_alpha (=1), SigmoidMin, SigmoidMax (=1), and SigmoidScale.
   */
  virtual void
  AutomaticParameterEstimation(void);

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

  PreconditionedGradientDescent(const Self &) = delete;
  void
  operator=(const Self &) = delete;

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
