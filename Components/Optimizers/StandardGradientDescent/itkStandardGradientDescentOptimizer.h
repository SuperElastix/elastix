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

#ifndef itkStandardGradientDescentOptimizer_h
#define itkStandardGradientDescentOptimizer_h

#include "itkGradientDescentOptimizer2.h"

namespace itk
{

/**
 * \class StandardGradientDescentOptimizer
 * \brief This class implements a gradient descent optimizer with a decaying gain.
 *
 * If \f$C(x)\f$ is a costfunction that has to be minimised, the following iterative
 * algorithm is used to find the optimal parameters \f$x\f$:
 *
 *   \f[ x(k+1) = x(k) - a(k) dC/dx \f]
 *
 * The gain \f$a(k)\f$ at each iteration \f$k\f$ is defined by:
 *
 *   \f[ a(k) =  a / (A + k + 1)^\alpha \f].
 *
 * It is very suitable to be used in combination with a stochastic estimate
 * of the gradient \f$dC/dx\f$. For example, in image registration problems it is
 * often advantageous to compute the metric derivative (\f$dC/dx\f$) on a new set
 * of randomly selected image samples in each iteration. You may set the parameter
 * \c NewSamplesEveryIteration to \c "true" to achieve this effect.
 * For more information on this strategy, you may have a look at:
 *
 * S. Klein, M. Staring, J.P.W. Pluim,
 * "Comparison of gradient approximation techniques for optimisation of mutual information in nonrigid registration",
 * in: SPIE Medical Imaging: Image Processing,
 * Editor(s): J.M. Fitzpatrick, J.M. Reinhardt, SPIE press, 2005, vol. 5747, Proceedings of SPIE, pp. 192-203.
 *
 * Or:
 *
 * S. Klein, M. Staring, J.P.W. Pluim,
 * "Evaluation of Optimization Methods for Nonrigid Medical Image Registration using Mutual Information and B-Splines"
 * IEEE Transactions on Image Processing, 2007, nr. 16(12), December.
 *
 * This class also serves as a base class for other GradientDescent type
 * algorithms, like the AcceleratedGradientDescentOptimizer.
 *
 * \sa StandardGradientDescent, AcceleratedGradientDescentOptimizer
 * \ingroup Optimizers
 */

class StandardGradientDescentOptimizer : public GradientDescentOptimizer2
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(StandardGradientDescentOptimizer);

  /** Standard ITK.*/
  using Self = StandardGradientDescentOptimizer;
  using Superclass = GradientDescentOptimizer2;

  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(StandardGradientDescentOptimizer, GradientDescentOptimizer2);

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
   * However, in inheriting classes, like the AcceleratedGradientDescent
   * the initial time may have a different function than Param_A.
   * Default: 0.0 */
  itkSetMacro(InitialTime, double);
  itkGetConstMacro(InitialTime, double);

  /** Get the current time. This equals the CurrentIteration in this base class
   * but may be different in inheriting classes, such as the AccelerateGradientDescent */
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
  StandardGradientDescentOptimizer();
  ~StandardGradientDescentOptimizer() override = default;

  /** Function to compute the parameter at time/iteration k. */
  virtual double
  Compute_a(double k) const;

  /** Function to update the current time
   * This function just increments the CurrentTime by 1.
   * Inheriting functions may implement something smarter,
   * for example, dependent on the progress */
  virtual void
  UpdateCurrentTime();

  /** The current time, which serves as input for Compute_a */
  double m_CurrentTime{ 0.0 };

  /** Constant step size or others, different value of k. */
  bool m_UseConstantStep{ false };

private:
  /**Parameters, as described by Spall.*/
  double m_Param_a{ 1.0 };
  double m_Param_A{ 1.0 };
  double m_Param_alpha{ 0.602 };

  /** Settings */
  double m_InitialTime{ 0.0 };
};

} // end namespace itk

#endif // end #ifndef itkStandardGradientDescentOptimizer_h
