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
#ifndef itkStandardStochasticVarianceReducedGradientOptimizer_h
#define itkStandardStochasticVarianceReducedGradientOptimizer_h

#include "itkStochasticVarianceReducedGradientDescentOptimizer.h"

namespace itk
{
/**
 * \class StandardStochasticVarianceReducedGradientOptimizer
 * \brief This class implements a gradient descent optimizer with a decaying gain.
 *
 * If \f$C(x)\f$ is a cost function that has to be minimised, the following iterative
 * algorithm is used to find the optimal parameters \f$x\f$:
 *
 *   \f[ x(k+1) = x(k) - a(k) dC/dx \f]
 *
 * The gain \f$a(k)\f$ at each iteration \f$k\f$ is defined by:
 *
 *   \f[ a(k) =  a / (A + k + 1)^alpha \f].
 *
 * It is very suitable to be used in combination with a stochastic estimate
 * of the gradient \f$dC/dx\f$. For example, in image registration problems it is
 * often advantageous to compute the metric derivative (\f$dC/dx\f$) on a new set
 * of randomly selected image samples in each iteration. You may set the parameter
 * \c NewSamplesEveryIteration to \c "true" to achieve this effect.
 * For more information on this strategy, you may have a look at:
 *
 * S. Klein, M. Staring, J.P.W. Pluim,
 * "Evaluation of Optimization Methods for Nonrigid Medical Image Registration using Mutual Information and B-Splines"
 * IEEE Transactions on Image Processing, 2007, nr. 16(12), December.
 *
 * This class also serves as a base class for other StochasticVarianceReducedGradient type
 * algorithms, like the AcceleratedStochasticVarianceReducedGradientOptimizer.
 *
 * \sa StandardStochasticVarianceReducedGradient, AcceleratedStochasticVarianceReducedGradientOptimizer
 * \ingroup Optimizers
 */

class StandardStochasticVarianceReducedGradientOptimizer : public StochasticVarianceReducedGradientDescentOptimizer
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(StandardStochasticVarianceReducedGradientOptimizer);

  /** Standard ITK.*/
  using Self = StandardStochasticVarianceReducedGradientOptimizer;
  using Superclass = StochasticVarianceReducedGradientDescentOptimizer;

  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(StandardStochasticVarianceReducedGradientOptimizer, StochasticVarianceReducedGradientDescentOptimizer);

  /** Typedefs inherited from the superclass. */
  using Superclass::MeasureType;
  using Superclass::ParametersType;
  using Superclass::DerivativeType;
  using Superclass::CostFunctionType;
  using Superclass::ScalesType;
  using Superclass::ScaledCostFunctionType;
  using Superclass::ScaledCostFunctionPointer;
  using Superclass::StopConditionType;

  /** Set/Get a. */
  itkSetMacro(Param_a, double);
  itkGetConstMacro(Param_a, double);

  /** Set/Get beta. */
  itkSetMacro(Param_beta, double);
  itkGetConstMacro(Param_beta, double);

  /** Set/Get A. */
  itkSetMacro(Param_A, double);
  itkGetConstMacro(Param_A, double);

  /** Set/Get alpha. */
  itkSetMacro(Param_alpha, double);
  itkGetConstMacro(Param_alpha, double);

  /** Sets a new LearningRate before calling the Superclass'
   * implementation, and updates the current time. */
  void
  AdvanceOneStep() override;

  /** Set current time to 0 and call superclass' implementation. */
  void
  StartOptimization() override;

  /** Set/Get the initial time. Should be >=0. This function is
   * superfluous, since Param_A does effectively the same.
   * However, in inheriting classes, like the AcceleratedStochasticVarianceReducedGradient
   * the initial time may have a different function than Param_A.
   * Default: 0.0 */
  itkSetMacro(InitialTime, double);
  itkGetConstMacro(InitialTime, double);

  /** Get the current time. This equals the CurrentIteration in this base class
   * but may be different in inheriting classes, such as the AccelerateStochasticVarianceReducedGradient */
  itkGetConstMacro(CurrentTime, double);

  /** Set the current time to the initial time. This can be useful
   * to 'reset' the optimisation, for example if you changed the
   * cost function while optimisation. Be careful with this function. */
  virtual void
  ResetCurrentTimeToInitialTime()
  {
    this->m_CurrentTime = this->m_InitialTime;
  }

protected:
  StandardStochasticVarianceReducedGradientOptimizer();
  ~StandardStochasticVarianceReducedGradientOptimizer() override = default;

  /** Function to compute the step size for SGD at time/iteration k. */
  virtual double
  Compute_a(double k) const;

  /** Function to compute the step size for SQN at time/iteration k. */
  virtual double
  Compute_beta(double k) const;

  /** Function to update the current time
   * This function just increments the CurrentTime by 1.
   * Inheriting functions may implement something smarter,
   * for example, dependent on the progress.
   */
  virtual void
  UpdateCurrentTime();

  /** The current time, which serves as input for Compute_a */
  double m_CurrentTime{ 0.0 };

  /** Constant step size or others, different value of k. */
  bool m_UseConstantStep;

private:
  /**Parameters, as described by Spall.*/
  double m_Param_a{ 1.0 };
  double m_Param_beta;
  double m_Param_A{ 1.0 };
  double m_Param_alpha{ 0.602 };

  /** Settings */
  double m_InitialTime{ 0.0 };

}; // end class StandardStochasticVarianceReducedGradientOptimizer


} // end namespace itk


#endif // end #ifndef itkStandardStochasticVarianceReducedGradientOptimizer_h
