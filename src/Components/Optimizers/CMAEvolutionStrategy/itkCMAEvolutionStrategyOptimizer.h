/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/


#ifndef __itkCMAEvolutionStrategyOptimizer_h
#define __itkCMAEvolutionStrategyOptimizer_h

#include "itkScaledSingleValuedNonLinearOptimizer.h"
#include <vector>
#include <utility>
#include <deque>

#include "itkArray.h"
#include "itkArray2D.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"
#include "vnl/vnl_diag_matrix.h"


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

  class CMAEvolutionStrategyOptimizer :
    public ScaledSingleValuedNonLinearOptimizer
  {
  public:

    typedef CMAEvolutionStrategyOptimizer     Self;
    typedef ScaledSingleValuedNonLinearOptimizer  Superclass;
    typedef SmartPointer<Self>                    Pointer;
    typedef SmartPointer<const Self>              ConstPointer;

    itkNewMacro(Self);
    itkTypeMacro(CMAEvolutionStrategyOptimizer,
      ScaledSingleValuedNonLinearOptimizer);

    typedef Superclass::ParametersType            ParametersType;
    typedef Superclass::DerivativeType            DerivativeType;
    typedef Superclass::CostFunctionType          CostFunctionType;
    typedef Superclass::ScaledCostFunctionType    ScaledCostFunctionType;
    typedef Superclass::MeasureType               MeasureType;
    typedef Superclass::ScalesType                ScalesType;

    typedef enum {
      MetricError,
      MaximumNumberOfIterations,
      PositionToleranceMin,
      PositionToleranceMax,
      ValueTolerance,
      ZeroStepLength,
      Unknown }                                   StopConditionType;

    virtual void StartOptimization(void);
    virtual void ResumeOptimization(void);
    virtual void StopOptimization(void);

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
    virtual double GetCurrentStepLength( void ) const
    { return this->GetCurrentSigma() * this->GetCurrentMaximumD();  }

    /** Get the last step taken ( scaledPos_{k+1} - scaledPos_{k} )
     * If you want the step taken in the space of unscaled parameters,
     * simply use:
     * CMAESOptimizer->GetScaledCostFunction()->ConvertScaledToUnscaledParameters( ... )
     * To obtain the magnitude of the step, use ->GetCurretScaledStep().magnitude().  */
    itkGetConstReferenceMacro( CurrentScaledStep, ParametersType );

    /** Setting: convergence condition: the maximum number of iterations. Default: 100 */
    itkGetConstMacro(MaximumNumberOfIterations, unsigned long);
    itkSetClampMacro(MaximumNumberOfIterations, unsigned long,
      1, NumericTraits<unsigned long>::max());

    /** Setting: the population size (\f$\lambda\f$);
     * if set to 0, a default value is chosen: 4 + floor( 3 ln(NumberOfParameters) ),
     * which  can be inspected after having started the optimisation.
     * Default: 0 */
    itkSetMacro( PopulationSize, unsigned int );
    itkGetConstMacro( PopulationSize, unsigned int );

    /** Setting: the number of parents (points for recombination, \f$\mu\f$)
     * if set to 0, a default value is chosen: floor( populationSize / 2 ),
     * which can be inspected after having started the optimisation.
     * Default: 0 */
    itkSetMacro( NumberOfParents, unsigned int );
    itkGetConstMacro( NumberOfParents, unsigned int );

    /** Setting: the initial standard deviation used to generate offspring
     * Recommended value: 1/3 * the expected range of the parameters
     * Default: 1.0;  */
    itkSetClampMacro( InitialSigma, double, NumericTraits<double>::min(), NumericTraits<double>::max() );
    itkGetConstMacro( InitialSigma, double );

    /** Setting: the maximum deviation. It is ensured that:
     * max_i( sigma*sqrt(C[i,i]) ) < MaximumDeviation
     * Default: +infinity */
    itkSetClampMacro( MaximumDeviation, double, 0.0, NumericTraits<double>::max() );
    itkGetConstMacro( MaximumDeviation, double );

    /** Setting: the minimum deviation. It is ensured that:
     * min_i( sigma*sqrt(C[i,i]) ) > MinimumDeviation
     * Default: 0.0 */
    itkSetClampMacro( MinimumDeviation, double, 0.0, NumericTraits<double>::max() );
    itkGetConstMacro( MinimumDeviation, double );

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
    itkGetConstMacro( UseDecayingSigma, bool );

    /** Setting: the A parameter for the decaying sigma sequence.
    * Default: 50 */
    itkSetClampMacro( SigmaDecayA, double, 0.0, NumericTraits<double>::max() );
    itkGetConstMacro( SigmaDecayA, double );

    /** Setting: the alpha parameter for the decaying sigma sequence.
     * Default: 0.602 */
    itkSetClampMacro( SigmaDecayAlpha, double, 0.0, 1.0 );
    itkGetConstMacro( SigmaDecayAlpha, double );

    /** Setting: whether the covariance matrix adaptation scheme should be used.
     * Default: true. If false: CovMatrix = Identity.
     * This parameter may be changed by the optimiser, if it sees that the
     * adaptation rate is nearly 0 (UpdateBDPeriod >= MaxNrOfIterations).
     * This can be inspected calling StartOptimization() */
    itkSetMacro( UseCovarianceMatrixAdaptation, bool );
    itkGetConstMacro( UseCovarianceMatrixAdaptation, bool );

    /** Setting: how the recombination weights are chosen:
     * "equal", "linear" or "superlinear" are supported
     * equal:       weights = ones(mu,1);
     * linear:      weights = mu+1-(1:mu)';
     * superlinear: weights = log(mu+1)-log(1:mu)';
     * Default: "superlinear" */
    itkSetStringMacro( RecombinationWeightsPreset );
    itkGetStringMacro( RecombinationWeightsPreset );

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
    itkSetMacro( PositionToleranceMin, double );
    itkGetConstMacro( PositionToleranceMin, double );

    /** Setting: convergence condition: the maximum step size.
     * 'convergence' is declared if:
     * if ( sigma * sqrt(C[i,i]) > PositionToleranceMax*sigma0   for any i )
     * Default: 1e8 */
    itkSetMacro( PositionToleranceMax, double );
    itkGetConstMacro( PositionToleranceMax, double );

    /** Setting: convergence condition: the minimum change of the cost function value over time.
     * convergence is declared if:
     * the range of the best cost function value measured over a period
     * of M iterations was not greater than the valueTolerance, where:
     * M = m_HistoryLength = min( maxnrofit, 10+ceil(3*10*N/lambda) ).
     * Default: 1e-12 */
    itkSetMacro( ValueTolerance, double );
    itkGetConstMacro( ValueTolerance, double );

  protected:

    typedef Array<double>                     RecombinationWeightsType;
    typedef vnl_diag_matrix<double>           EigenValueMatrixType;
    typedef Array2D<double>                   CovarianceMatrixType;
    typedef std::vector< ParametersType >     ParameterContainerType;
    typedef std::deque< MeasureType >         MeasureHistoryType;

    typedef
      std::pair< MeasureType, unsigned int >  MeasureIndexPairType;
    typedef std::vector<MeasureIndexPairType> MeasureContainerType;

    typedef itk::Statistics::MersenneTwisterRandomVariateGenerator RandomGeneratorType;

    /** The random number generator used to generate the offspring. */
    RandomGeneratorType::Pointer m_RandomGenerator;

    /** The value of the cost function at the current position */
    MeasureType                   m_CurrentValue;

    /** The current iteration number */
    unsigned long                 m_CurrentIteration;

    /** The stop condition */
    StopConditionType             m_StopCondition;

    /** Boolean that indicates whether the optimizer should stop */
    bool                          m_Stop;

    /** Settings that may be changed by the optimizer: */
    bool                          m_UseCovarianceMatrixAdaptation;
    unsigned int                  m_PopulationSize;
    unsigned int                  m_NumberOfParents;
    unsigned int                  m_UpdateBDPeriod;

    /** Some other constants, without set/get methods
     * These settings have default values. */

    /** \f$\mu_{eff}\f$ */
    double                        m_EffectiveMu;
    /** \f$c_{\sigma}\f$ */
    double                        m_ConjugateEvolutionPathConstant;
    /** \f$d_{\sigma}\f$ */
    double                        m_SigmaDampingConstant;
    /** \f$c_{cov}\f$ */
    double                        m_CovarianceMatrixAdaptationConstant;
    /** \f$c_c\f$ */
    double                        m_EvolutionPathConstant;
    /** \f$\mu_{cov} = \mu_{eff}\f$ */
    double                        m_CovarianceMatrixAdaptationWeight;
    /** \f$chiN  = E( \|N(0,I)\|\f$ */
    double                        m_ExpectationNormNormalDistribution;
    /** array of \f$w_i\f$ */
    RecombinationWeightsType      m_RecombinationWeights;
    /** Length of the MeasureHistory deque */
    unsigned long                 m_HistoryLength;

    /** The current value of Sigma */
    double                        m_CurrentSigma;

    /** The current minimum square root eigen value: */
    double                        m_CurrentMinimumD;
    /** The current maximum square root eigen value: */
    double                        m_CurrentMaximumD;

    /** \f$h_{\sigma}\f$ */
    bool                          m_Heaviside;

    /** \f$d_i = x_i - m\f$ */
    ParameterContainerType        m_SearchDirs;
    /** realisations of \f$N(0,I)\f$ */
    ParameterContainerType        m_NormalizedSearchDirs;
    /** cost function values for each \f$x_i = m + d_i\f$ */
    MeasureContainerType          m_CostFunctionValues;
    /** \f$m(g+1) - m(g)\f$ */
    ParametersType                m_CurrentScaledStep;
    /** \f$1/\sigma * D^{-1} * B' * m_CurrentScaledStep, needed for p_{\sigma}\f$ */
    ParametersType                m_CurrentNormalizedStep;
    /** \f$p_c\f$ */
    ParametersType                m_EvolutionPath;
    /** \f$p_\sigma\f$ */
    ParametersType                m_ConjugateEvolutionPath;

    /** History of best measure values */
    MeasureHistoryType            m_MeasureHistory;

    /** C: covariance matrix */
    CovarianceMatrixType          m_C;
    /** B: eigen vector matrix */
    CovarianceMatrixType          m_B;
    /** D: sqrt(eigen values) */
    EigenValueMatrixType          m_D;

    /** Constructor */
    CMAEvolutionStrategyOptimizer();

    /** Destructor */
    virtual ~CMAEvolutionStrategyOptimizer(){};

    /** PrintSelf */
    void PrintSelf(std::ostream& os, Indent indent) const;

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
    virtual void InitializeConstants(void);

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
    virtual void InitializeProgressVariables(void);

    /** Initialize the covariance matrix and its eigen decomposition */
    virtual void InitializeBCD(void);

    /** GenerateOffspring: Fill m_SearchDirs, m_NormalizedSearchDirs,
     * and m_CostFunctionValues */
    virtual void GenerateOffspring(void);

    /** Sort the m_CostFunctionValues vector and update m_MeasureHistory */
    virtual void SortCostFunctionValues(void);

    /** Compute the m_CurrentPosition = m(g+1), m_CurrentValue, and m_CurrentScaledStep */
    virtual void AdvanceOneStep(void);

    /** Update m_ConjugateEvolutionPath */
    virtual void UpdateConjugateEvolutionPath(void);

    /** Update m_Heaviside */
    virtual void UpdateHeaviside( void );

    /** Update m_EvolutionPath */
    virtual void UpdateEvolutionPath(void);

    /** Update the covariance matrix C */
    virtual void UpdateC(void);

    /** Update the Sigma either by adaptation or using the predefined function */
    virtual void UpdateSigma(void);

    /** Update the eigen decomposition and m_CurrentMaximumD/m_CurrentMinimumD */
    virtual void UpdateBD(void);

    /** Some checks, to be sure no numerical errors occur
     * \li Adjust too low/high deviation that otherwise would violate
     * m_MinimumDeviation or m_MaximumDeviation.
     * \li Adjust too low deviations that otherwise would cause numerical
     * problems (because of finite precision of the datatypes).
     * \li Check if "main axis standard deviation sigma*D(i,i) has effect" (?)
     * (just another check whether the steps are not too small)
     * \li Adjust step size in case of equal function values (flat fitness)
     * \li Adjust step size in case of equal best function values over history  */
    virtual void FixNumericalErrors(void);

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
    virtual bool TestConvergence( bool firstCheck );

  private:
    CMAEvolutionStrategyOptimizer(const Self&); // purposely not implemented
    void operator=(const Self&); // purposely not implemented

    /** Settings that are only inspected/changed by the associated get/set member functions. */
    unsigned long                 m_MaximumNumberOfIterations;
    bool                          m_UseDecayingSigma;
    double                        m_InitialSigma;
    double                        m_SigmaDecayA;
    double                        m_SigmaDecayAlpha;
    std::string                   m_RecombinationWeightsPreset;
    double                        m_MaximumDeviation;
    double                        m_MinimumDeviation;
    double                        m_PositionToleranceMax;
    double                        m_PositionToleranceMin;
    double                        m_ValueTolerance;

  };


} // end namespace itk


#endif //#ifndef __itkCMAEvolutionStrategyOptimizer_h

