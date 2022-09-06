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

#ifndef itkCMAEvolutionStrategyOptimizer_h
#define itkCMAEvolutionStrategyOptimizer_h

#include "itkScaledSingleValuedNonLinearOptimizer.h"
#include <vector>
#include <utility>
#include <deque>

#include "itkArray.h"
#include "itkArray2D.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"
#include <vnl/vnl_diag_matrix.h>

namespace itk
{
/**
 * \class CMAEvolutionStrategyOptimizer
 * \brief A Covariance Matrix Adaptation Evolution Strategy Optimizer.
 *
 * Based on the work by Hansen:
 *   - http://www.bionik.tu-berlin.de/user/niko/
 *   - Hansen and Ostermeier,
 *     "Completely Derandomized Self-Adaptation in Evolution Strategies",
 *     Evolutionary Computation, 9(2), pp. 159-195 (2001).
 *   - See also the Matlab code, cmaes.m, which you can download from the
 *     website mentioned above.
 *
 * \ingroup Numerics Optimizers
 */

class CMAEvolutionStrategyOptimizer : public ScaledSingleValuedNonLinearOptimizer
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(CMAEvolutionStrategyOptimizer);

  using Self = CMAEvolutionStrategyOptimizer;
  using Superclass = ScaledSingleValuedNonLinearOptimizer;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  itkNewMacro(Self);
  itkTypeMacro(CMAEvolutionStrategyOptimizer, ScaledSingleValuedNonLinearOptimizer);

  using Superclass::ParametersType;
  using Superclass::DerivativeType;
  using Superclass::CostFunctionType;
  using Superclass::ScaledCostFunctionType;
  using Superclass::MeasureType;
  using Superclass::ScalesType;

  enum StopConditionType
  {
    MetricError,
    MaximumNumberOfIterations,
    PositionToleranceMin,
    PositionToleranceMax,
    ValueTolerance,
    ZeroStepLength,
    Unknown
  };

  void
  StartOptimization() override;

  virtual void
  ResumeOptimization();

  virtual void
  StopOptimization();

  /** Get the current iteration number: */
  itkGetConstMacro(CurrentIteration, unsigned long);

  /** Get the metric value at the current position */
  itkGetConstMacro(CurrentValue, MeasureType);

  /** Get the stop condition of the last run */
  itkGetConstReferenceMacro(StopCondition, StopConditionType);

  /** The current value of sigma */
  itkGetConstMacro(CurrentSigma, double);

  /** The current minimum square root eigen value */
  itkGetConstMacro(CurrentMinimumD, double);

  /** The current maximum square root eigen value */
  itkGetConstMacro(CurrentMaximumD, double);

  /** This function is just for convenience, since many optimizers have such
   * a function. It return the current sigma times the current maximumD. */
  virtual double
  GetCurrentStepLength() const
  {
    return this->GetCurrentSigma() * this->GetCurrentMaximumD();
  }

  /** Get the last step taken ( scaledPos_{k+1} - scaledPos_{k} )
   * If you want the step taken in the space of unscaled parameters,
   * simply use:
   * CMAESOptimizer->GetScaledCostFunction()->ConvertScaledToUnscaledParameters( ... )
   * To obtain the magnitude of the step, use ->GetCurretScaledStep().magnitude().  */
  itkGetConstReferenceMacro(CurrentScaledStep, ParametersType);

  /** Setting: convergence condition: the maximum number of iterations. Default: 100 */
  itkGetConstMacro(MaximumNumberOfIterations, unsigned long);
  itkSetClampMacro(MaximumNumberOfIterations, unsigned long, 1, NumericTraits<unsigned long>::max());

  /** Setting: the population size (\f$\lambda\f$);
   * if set to 0, a default value is chosen: 4 + floor( 3 ln(NumberOfParameters) ),
   * which  can be inspected after having started the optimisation.
   * Default: 0 */
  itkSetMacro(PopulationSize, unsigned int);
  itkGetConstMacro(PopulationSize, unsigned int);

  /** Setting: the number of parents (points for recombination, \f$\mu\f$)
   * if set to 0, a default value is chosen: floor( populationSize / 2 ),
   * which can be inspected after having started the optimisation.
   * Default: 0 */
  itkSetMacro(NumberOfParents, unsigned int);
  itkGetConstMacro(NumberOfParents, unsigned int);

  /** Setting: the initial standard deviation used to generate offspring
   * Recommended value: 1/3 * the expected range of the parameters
   * Default: 1.0;  */
  itkSetClampMacro(InitialSigma, double, NumericTraits<double>::min(), NumericTraits<double>::max());
  itkGetConstMacro(InitialSigma, double);

  /** Setting: the maximum deviation. It is ensured that:
   * max_i( sigma*sqrt(C[i,i]) ) < MaximumDeviation
   * Default: +infinity */
  itkSetClampMacro(MaximumDeviation, double, 0.0, NumericTraits<double>::max());
  itkGetConstMacro(MaximumDeviation, double);

  /** Setting: the minimum deviation. It is ensured that:
   * min_i( sigma*sqrt(C[i,i]) ) > MinimumDeviation
   * Default: 0.0 */
  itkSetClampMacro(MinimumDeviation, double, 0.0, NumericTraits<double>::max());
  itkGetConstMacro(MinimumDeviation, double);

  /** Setting: Use a sigma that decays according to a predefined function,
   * instead of the adaptive scheme proposed by Hansen et al.
   * if true: currentsigma(k+1) = currentsigma(k) * (A+k)^alpha / (A+k+1)^alpha
   * where:
   * k = the current iteration
   * A, alpha = user-specified parameters (see below)
   *
   * Default: false
   */
  itkSetMacro(UseDecayingSigma, bool);
  itkGetConstMacro(UseDecayingSigma, bool);

  /** Setting: the A parameter for the decaying sigma sequence.
   * Default: 50 */
  itkSetClampMacro(SigmaDecayA, double, 0.0, NumericTraits<double>::max());
  itkGetConstMacro(SigmaDecayA, double);

  /** Setting: the alpha parameter for the decaying sigma sequence.
   * Default: 0.602 */
  itkSetClampMacro(SigmaDecayAlpha, double, 0.0, 1.0);
  itkGetConstMacro(SigmaDecayAlpha, double);

  /** Setting: whether the covariance matrix adaptation scheme should be used.
   * Default: true. If false: CovMatrix = Identity.
   * This parameter may be changed by the optimiser, if it sees that the
   * adaptation rate is nearly 0 (UpdateBDPeriod >= MaxNrOfIterations).
   * This can be inspected calling StartOptimization() */
  itkSetMacro(UseCovarianceMatrixAdaptation, bool);
  itkGetConstMacro(UseCovarianceMatrixAdaptation, bool);

  /** Setting: how the recombination weights are chosen:
   * "equal", "linear" or "superlinear" are supported
   * equal:       weights = ones(mu,1);
   * linear:      weights = mu+1-(1:mu)';
   * superlinear: weights = log(mu+1)-log(1:mu)';
   * Default: "superlinear" */
  itkSetStringMacro(RecombinationWeightsPreset);
  itkGetStringMacro(RecombinationWeightsPreset);

  /** Setting: the number of iterations after which B and D are updated.
   * If 0: a default value is computed: floor( 1.0 / c_cov / Nd / 10.0 )
   * This value can be inspected after calling StartOptimization  */
  itkSetMacro(UpdateBDPeriod, unsigned int);
  itkGetConstMacro(UpdateBDPeriod, unsigned int);

  /** Setting: convergence condition: the minimum step size.
   * convergence is declared if:
   * if ( sigma * max( abs(p_c[i]), sqrt(C[i,i]) ) < PositionToleranceMin*sigma0  for all i )
   * where p_c is the evolution path
   * Default: 1e-12 */
  itkSetMacro(PositionToleranceMin, double);
  itkGetConstMacro(PositionToleranceMin, double);

  /** Setting: convergence condition: the maximum step size.
   * 'convergence' is declared if:
   * if ( sigma * sqrt(C[i,i]) > PositionToleranceMax*sigma0   for any i )
   * Default: 1e8 */
  itkSetMacro(PositionToleranceMax, double);
  itkGetConstMacro(PositionToleranceMax, double);

  /** Setting: convergence condition: the minimum change of the cost function value over time.
   * convergence is declared if:
   * the range of the best cost function value measured over a period
   * of M iterations was not greater than the valueTolerance, where:
   * M = m_HistoryLength = min( maxnrofit, 10+ceil(3*10*N/lambda) ).
   * Default: 1e-12 */
  itkSetMacro(ValueTolerance, double);
  itkGetConstMacro(ValueTolerance, double);

protected:
  using RecombinationWeightsType = Array<double>;
  using EigenValueMatrixType = vnl_diag_matrix<double>;
  using CovarianceMatrixType = Array2D<double>;
  using ParameterContainerType = std::vector<ParametersType>;
  using MeasureHistoryType = std::deque<MeasureType>;

  using MeasureIndexPairType = std::pair<MeasureType, unsigned int>;
  using MeasureContainerType = std::vector<MeasureIndexPairType>;

  using RandomGeneratorType = itk::Statistics::MersenneTwisterRandomVariateGenerator;

  /** The random number generator used to generate the offspring. */
  RandomGeneratorType::Pointer m_RandomGenerator{ RandomGeneratorType::GetInstance() };

  /** The value of the cost function at the current position */
  MeasureType m_CurrentValue{ 0.0 };

  /** The current iteration number */
  unsigned long m_CurrentIteration{ 0 };

  /** The stop condition */
  StopConditionType m_StopCondition{ Unknown };

  /** Boolean that indicates whether the optimizer should stop */
  bool m_Stop{ false };

  /** Settings that may be changed by the optimizer: */
  bool         m_UseCovarianceMatrixAdaptation{ true };
  unsigned int m_PopulationSize{ 0 };
  unsigned int m_NumberOfParents{ 0 };
  unsigned int m_UpdateBDPeriod{ 1 };

  /** Some other constants, without set/get methods
   * These settings have default values. */

  /** \f$\mu_{eff}\f$ */
  double m_EffectiveMu{ 0.0 };
  /** \f$c_{\sigma}\f$ */
  double m_ConjugateEvolutionPathConstant{ 0.0 };
  /** \f$d_{\sigma}\f$ */
  double m_SigmaDampingConstant{ 0.0 };
  /** \f$c_{cov}\f$ */
  double m_CovarianceMatrixAdaptationConstant{ 0.0 };
  /** \f$c_c\f$ */
  double m_EvolutionPathConstant{ 0.0 };
  /** \f$\mu_{cov} = \mu_{eff}\f$ */
  double m_CovarianceMatrixAdaptationWeight{ 0.0 };
  /** \f$chiN  = E( \|N(0,I)\|\f$ */
  double m_ExpectationNormNormalDistribution{ 0.0 };
  /** array of \f$w_i\f$ */
  RecombinationWeightsType m_RecombinationWeights;
  /** Length of the MeasureHistory deque */
  unsigned long m_HistoryLength{ 0 };

  /** The current value of Sigma */
  double m_CurrentSigma{ 0.0 };

  /** The current minimum square root eigen value: */
  double m_CurrentMinimumD{ 1.0 };
  /** The current maximum square root eigen value: */
  double m_CurrentMaximumD{ 1.0 };

  /** \f$h_{\sigma}\f$ */
  bool m_Heaviside{ false };

  /** \f$d_i = x_i - m\f$ */
  ParameterContainerType m_SearchDirs;
  /** realisations of \f$N(0,I)\f$ */
  ParameterContainerType m_NormalizedSearchDirs;
  /** cost function values for each \f$x_i = m + d_i\f$ */
  MeasureContainerType m_CostFunctionValues;
  /** \f$m(g+1) - m(g)\f$ */
  ParametersType m_CurrentScaledStep;
  /** \f$1/\sigma * D^{-1} * B' * m_CurrentScaledStep, needed for p_{\sigma}\f$ */
  ParametersType m_CurrentNormalizedStep;
  /** \f$p_c\f$ */
  ParametersType m_EvolutionPath;
  /** \f$p_\sigma\f$ */
  ParametersType m_ConjugateEvolutionPath;

  /** History of best measure values */
  MeasureHistoryType m_MeasureHistory;

  /** C: covariance matrix */
  CovarianceMatrixType m_C;
  /** B: eigen vector matrix */
  CovarianceMatrixType m_B;
  /** D: sqrt(eigen values) */
  EigenValueMatrixType m_D;

  /** Constructor */
  CMAEvolutionStrategyOptimizer();

  /** Destructor */
  ~CMAEvolutionStrategyOptimizer() override = default;

  /** PrintSelf */
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Compute the following constant variables:
   * \li m_PopulationSize (if not provided by the user)
   * \li m_NumberOfParents (if not provided by the user)
   * \li m_RecombinationWeights
   * \li m_EffectiveMu
   * \li m_ConjugateEvolutionPathConstant
   * \li m_SigmaDampingConstant
   * \li m_CovarianceMatrixAdaptationWeight
   * \li m_CovarianceMatrixAdaptationConstant
   * \li m_UpdateBDPeriod;
   * \li m_UseCovarianceMatrixAdaptation;
   * \li m_EvolutionPathConstant
   * \li m_ExpectationNormNormalDistribution
   * \li m_HistoryLength */
  virtual void
  InitializeConstants();

  /** Initialize the following 'progress' variables:
   * \li m_CurrentSigma
   * \li m_Heaviside
   * \li m_SearchDirs
   * \li m_NormalizedSearchDirs
   * \li m_CostFunctionValues
   * \li m_CurrentScaledStep
   * \li m_CurrentNormalizedStep
   * \li m_EvolutionPath
   * \li m_ConjugateEvolutionPath
   * \li m_MeasureHistory
   * \li m_CurrentMaximumD, m_CurrentMinimumD */
  virtual void
  InitializeProgressVariables();

  /** Initialize the covariance matrix and its eigen decomposition */
  virtual void
  InitializeBCD();

  /** GenerateOffspring: Fill m_SearchDirs, m_NormalizedSearchDirs,
   * and m_CostFunctionValues */
  virtual void
  GenerateOffspring();

  /** Sort the m_CostFunctionValues vector and update m_MeasureHistory */
  virtual void
  SortCostFunctionValues();

  /** Compute the m_CurrentPosition = m(g+1), m_CurrentValue, and m_CurrentScaledStep */
  virtual void
  AdvanceOneStep();

  /** Update m_ConjugateEvolutionPath */
  virtual void
  UpdateConjugateEvolutionPath();

  /** Update m_Heaviside */
  virtual void
  UpdateHeaviside();

  /** Update m_EvolutionPath */
  virtual void
  UpdateEvolutionPath();

  /** Update the covariance matrix C */
  virtual void
  UpdateC();

  /** Update the Sigma either by adaptation or using the predefined function */
  virtual void
  UpdateSigma();

  /** Update the eigen decomposition and m_CurrentMaximumD/m_CurrentMinimumD */
  virtual void
  UpdateBD();

  /** Some checks, to be sure no numerical errors occur
   * \li Adjust too low/high deviation that otherwise would violate
   * m_MinimumDeviation or m_MaximumDeviation.
   * \li Adjust too low deviations that otherwise would cause numerical
   * problems (because of finite precision of the datatypes).
   * \li Check if "main axis standard deviation sigma*D(i,i) has effect" (?)
   * (just another check whether the steps are not too small)
   * \li Adjust step size in case of equal function values (flat fitness)
   * \li Adjust step size in case of equal best function values over history  */
  virtual void
  FixNumericalErrors();

  /** Check if convergence has occured:
   * \li Check if the maximum number of iterations will not be exceeded in the following iteration
   * \li Check if the step was not too large:
   *     if ( sigma * sqrt(C[i,i]) > PositionToleranceMax*sigma0   for any i )
   * \li Check for zero steplength (should never happen):
   *     if ( sigma * D[i] <= 0  for all i  )
   * \li if firstCheck==true -> quit function
   * \li Check if the step was not too small:
   *     if ( sigma * max( abs(p_c[i]), sqrt(C[i,i]) ) < PositionToleranceMin*sigma0  for all i )
   * \li Check if the value tolerance is satisfied.  */
  virtual bool
  TestConvergence(bool firstCheck);

private:
  /** Settings that are only inspected/changed by the associated get/set member functions. */
  unsigned long m_MaximumNumberOfIterations{ 100 };
  bool          m_UseDecayingSigma{ false };
  double        m_InitialSigma{ 1.0 };
  double        m_SigmaDecayA{ 50 };
  double        m_SigmaDecayAlpha{ 0.602 };
  std::string   m_RecombinationWeightsPreset{ "superlinear" };
  double        m_MaximumDeviation{ std::numeric_limits<double>::max() };
  double        m_MinimumDeviation{ 0.0 };
  double        m_PositionToleranceMax{ 1e8 };
  double        m_PositionToleranceMin{ 1e-12 };
  double        m_ValueTolerance{ 1e-12 };
};

} // end namespace itk

#endif //#ifndef itkCMAEvolutionStrategyOptimizer_h
