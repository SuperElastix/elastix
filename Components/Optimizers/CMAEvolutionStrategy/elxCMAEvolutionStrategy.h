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
#ifndef elxCMAEvolutionStrategy_h
#define elxCMAEvolutionStrategy_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkCMAEvolutionStrategyOptimizer.h"

namespace elastix
{

/**
 * \class CMAEvolutionStrategy
 * \brief An optimizer based on the itk::CMAEvolutionStrategyOptimizer.
 *
 * A Covariance-Matrix-Adaptation-Evolution-Strategy optimizer.
 *
 * This optimizer support the NewSamplesEveryIteration option. It requests
 * new samples for the computation of each search direction (not during
 * the offspring generation). The theory doesn't say anything about such a
 * situation, so, think twice before using the NewSamplesEveryIteration option.
 *
 * The parameters used in this class are:
 * \parameter Optimizer: Select this optimizer as follows:\n
 *    <tt>(Optimizer "CMAEvolutionStrategy")</tt>
 * \parameter MaximumNumberOfIterations: The maximum number of iterations in each resolution. \n
 *    example: <tt>(MaximumNumberOfIterations 100 100 50)</tt> \n
 *    Default value: 500.\n
 * \parameter StepLength: Set the length of the initial step ( = Sigma0 = InitialSigma).\n
 *    example: <tt>(StepLength 2.0 1.0 0.5)</tt> \n
 *    Recommended value: 1/3 of the expected parameter range.\n
 *    Default value: 1.0.\n
 * \parameter ValueTolerance: Stopping criterion. See the documentation of the
 *    itk::CMAEvolutionStrategyOptimizer for more information.\n
 *    example: <tt>(ValueTolerance 0.001 0.0001 0.000001)</tt> \n
 *    Default value: 0.00001. Can be specified for each resolution.\n
 * \parameter PositionToleranceMin: Stopping criterion. See the documentation of the
 *    itk::CMAEvolutionStrategyOptimizer for more information.\n
 *    example: <tt>(PositionToleranceMin 0.001 0.0001 0.000001)</tt> \n
 *    Default value: 1e-8. Can be specified for each resolution.\n
 * \parameter PositionToleranceMax: Stopping criterion. See the documentation of the
 *    itk::CMAEvolutionStrategyOptimizer for more information.\n
 *    example: <tt>(PositionToleranceMax 0.001 0.0001 0.000001)</tt> \n
 *    Default value: 1e8. Can be specified for each resolution.\n
 * \parameter PopulationSize: the number of parameter vectors evaluated in each iteration.\n
 *    If you set it to 0, a default value is calculated by the optimizer, which is reported
 *    back to the elastix.log file.\n
 *    example: <tt>(PopulationSize 0 20 20)</tt> \n
 *    Default: 0 (so, automatically determined). Can be specified for each resolution.\n
 * \parameter NumberOfParents: the number of parameter vectors selected for recombination.\n
 *    If you set it to 0, a default value is calculated
 *    by the optimizer, which is reported back to the elastix.log file.\n
 *    example: <tt>(NumberOfParents 0 10 10)</tt> \n
 *    Default: 0 (so, automatically determined). Can be specified for each resolution. \n
 *    This value must be less than or equal to the PopulationSize.\n
 * \parameter MaximumDeviation: the step length is limited to this value. See the documentation of the
 *    itk::CMAEvolutionStrategyOptimizer for more information.\n
 *    example: <tt>(MaximumDeviation 10.0 10.0 5.0)</tt> \n
 *    Default: 10.0 * positionToleranceMax = practically infinity. Can be specified for each resolution.\n
 * \parameter MinimumDeviation: the step length is ensured to be greater than this value.\n
 *    See the documentation of the itk::CMAEvolutionStrategyOptimizer for more information.\n
 *    example: <tt>(MinimumDeviation 0.01 0.01 0.0001)</tt> \n
 *    Default: 0.0. Can be specified for each resolution.\n
 * \parameter UseDecayingSigma: use a predefined decaying function to control the steplength sigma.\n
 *    example: <tt>(UseDecayingSigma "false" "true" "false")</tt> \n
 *    Default/recommended: "false". Can be specified for each resolution.\n
 *    If you set it to true the SP_A and SP_alpha parameters apply.\n
 * \parameter SP_A: If UseDecayingSigma is set to "true", the steplength \f$sigma(k)\f$ at each
 *    iteration \f$k\f$ is defined by: \n
 *    \f$sigma(k+1) = sigma(k) (SP\_A + k)^{SP\_alpha} / (SP\_A + k + 1)^{SP\_alpha}\f$. \n
 *    where sigma(0) is set by the parameter "StepLength". \n
 *    example: <tt>(SP_A 50.0 50.0 100.0)</tt> \n
 *    The default value is 50.0. SP_A can be defined for each resolution. \n
 * \parameter SP_alpha: If UseDecayingSigma is set to "true", the steplength \f$sigma(k)\f$ at each
 *    iteration \f$k\f$ is defined by: \n
 *    \f$sigma(k+1) = sigma(k) (SP\_A + k)^{SP\_alpha} / (SP\_A + k + 1)^{SP\_alpha}\f$. \n
 *    where sigma(0) is set by the parameter "StepLength".\n
 *    example: <tt>(SP_alpha 0.602 0.602 0.602)</tt> \n
 *    The default value is 0.602. SP_alpha can be defined for each resolution. \n
 * \parameter UseCovarianceMatrixAdaptation: a boolean that determines whether to use the
 *    covariance matrix adaptation scheme.\n
 *    example: <tt>(UseCovarianceMatrixAdaptation "false" "true" "true")</tt> \n
 *    Default: "true". This parameter may be altered by the optimizer. The actual value used is \n
 *    reported back in the elastix.log file. This parameter can be specified for each resolution. \n
 * \parameter RecombinationWeightsPreset: the name of a preset for the recombination weights.\n
 *    See the documentation of the itk::CMAEvolutionStrategyOptimizer for more information.\n
 *    example: <tt>(UseCovarianceMatrixAdaptation "equal" "linear" "superlinear")</tt> \n
 *    Default/recommended: "superlinear". Choose one of {"equal", "linear", "superlinear"}.
 *    This parameter can be specified for each resolution. \n
 * \parameter UpdateBDPeriod: the number of iterations after which the eigendecomposition of the
 *    covariance matrix is updated. If 0, the optimizer estimates a value. The actual value used is
 *    reported back in the elastix.log file. This parameter can be specified for each resolution. \n
 *    example: <tt>(UpdateBDPeriod 0 0 50)</tt> \n
 *    Default: 0 (so, automatically determined).
 *
 * \ingroup Optimizers
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT CMAEvolutionStrategy
  : public itk::CMAEvolutionStrategyOptimizer
  , public OptimizerBase<TElastix>
{
public:
  /** Standard ITK.*/
  typedef CMAEvolutionStrategy          Self;
  typedef CMAEvolutionStrategyOptimizer Superclass1;
  typedef OptimizerBase<TElastix>       Superclass2;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CMAEvolutionStrategy, CMAEvolutionStrategyOptimizer);

  /** Name of this class.
   * Use this name in the parameter file to select this specific optimizer. \n
   * example: <tt>(Optimizer "CMAEvolutionStrategy")</tt>\n
   */
  elxClassNameMacro("CMAEvolutionStrategy");

  /** Typedef's inherited from Superclass1.*/
  typedef Superclass1::CostFunctionType    CostFunctionType;
  typedef Superclass1::CostFunctionPointer CostFunctionPointer;
  typedef Superclass1::StopConditionType   StopConditionType;
  typedef Superclass1::ParametersType      ParametersType;
  typedef Superclass1::DerivativeType      DerivativeType;
  typedef Superclass1::ScalesType          ScalesType;

  /** Typedef's inherited from Elastix.*/
  typedef typename Superclass2::ElastixType          ElastixType;
  typedef typename Superclass2::ElastixPointer       ElastixPointer;
  typedef typename Superclass2::ConfigurationType    ConfigurationType;
  typedef typename Superclass2::ConfigurationPointer ConfigurationPointer;
  typedef typename Superclass2::RegistrationType     RegistrationType;
  typedef typename Superclass2::RegistrationPointer  RegistrationPointer;
  typedef typename Superclass2::ITKBaseType          ITKBaseType;

  /** Check if any scales are set, and set the UseScales flag on or off;
   * after that call the superclass' implementation */
  void
  StartOptimization(void) override;

  /** Methods to set parameters and print output at different stages
   * in the registration process.*/
  void
  BeforeRegistration(void) override;

  void
  BeforeEachResolution(void) override;

  void
  AfterEachResolution(void) override;

  void
  AfterEachIteration(void) override;

  void
  AfterRegistration(void) override;

protected:
  CMAEvolutionStrategy() = default;
  ~CMAEvolutionStrategy() override = default;

  /** Call the superclass' implementation and print the value of some variables */
  void
  InitializeProgressVariables(void) override;

private:
  elxOverrideGetSelfMacro;

  CMAEvolutionStrategy(const Self &) = delete;
  void
  operator=(const Self &) = delete;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxCMAEvolutionStrategy.hxx"
#endif

#endif // end #ifndef elxCMAEvolutionStrategy_h
