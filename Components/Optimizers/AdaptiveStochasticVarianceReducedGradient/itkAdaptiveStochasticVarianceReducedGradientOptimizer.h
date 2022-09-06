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
#ifndef itkAdaptiveStochasticVarianceReducedGradientOptimizer_h
#define itkAdaptiveStochasticVarianceReducedGradientOptimizer_h

#include "itkStandardStochasticVarianceReducedGradientDescentOptimizer.h"

namespace itk
{
/**
 * \class AdaptiveStochasticVarianceReducedGradientOptimizer
 * \brief This class implements a gradient descent optimizer with adaptive gain.
 *
 * If \f$C(x)\f$ is a costfunction that has to be minimised, the following iterative
 * algorithm is used to find the optimal parameters \f$x\f$:
 *
 *   \f[ x(k+1) = x(k) - a(t_k) dC/dx \f]
 *
 * The gain \f$a(t_k)\f$ at each iteration \f$k\f$ is defined by:
 *
 *   \f[ a(t_k) =  a / (A + t_k + 1)^alpha \f].
 *
 * And the time \f$t_k\f$ is updated according to:
 *
 *   \f[ t_{k+1} = [ t_k + sigmoid( -g_k^T g_{k-1} ) ]^+ \f]
 *
 * where \f$g_k\f$ equals \f$dC/dx\f$ at iteration \f$k\f$.
 * For \f$t_0\f$ the InitialTime is used, which is defined in the
 * the superclass (StandardGradientDescentOptimizer). Whereas in the
 * superclass this parameter is superfluous, in this class it makes sense.
 *
 * This method is described in the following references:
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
 * It is very suitable to be used in combination with a stochastic estimate
 * of the gradient \f$dC/dx\f$. For example, in image registration problems it is
 * often advantageous to compute the metric derivative (\f$dC/dx\f$) on a new set
 * of randomly selected image samples in each iteration. You may set the parameter
 * \c NewSamplesEveryIteration to \c "true" to achieve this effect.
 * For more information on this strategy, you may have a look at:
 *
 * \sa AdaptiveStochasticVarianceReducedGradient, StandardGradientDescentOptimizer
 * \ingroup Optimizers
 */

class AdaptiveStochasticVarianceReducedGradientOptimizer : public StandardStochasticVarianceReducedGradientOptimizer
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(AdaptiveStochasticVarianceReducedGradientOptimizer);

  /** Standard ITK.*/
  using Self = AdaptiveStochasticVarianceReducedGradientOptimizer;
  using Superclass = StandardStochasticVarianceReducedGradientOptimizer;

  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(AdaptiveStochasticVarianceReducedGradientOptimizer, StandardStochasticVarianceReducedGradientOptimizer);

  /** Typedefs inherited from the superclass. */
  using Superclass::MeasureType;
  using Superclass::ParametersType;
  using Superclass::DerivativeType;
  using Superclass::CostFunctionType;
  using Superclass::ScalesType;
  using Superclass::ScaledCostFunctionType;
  using Superclass::ScaledCostFunctionPointer;
  using Superclass::StopConditionType;

  /** Set/Get whether the adaptive step size mechanism is desired. Default: true */
  itkSetMacro(UseAdaptiveStepSizes, bool);
  itkGetConstMacro(UseAdaptiveStepSizes, bool);

  /** Set/Get the maximum of the sigmoid.
   * Should be >0. Default: 1.0 */
  itkSetMacro(SigmoidMax, double);
  itkGetConstMacro(SigmoidMax, double);

  /** Set/Get the maximum of the sigmoid.
   * Should be <0. Default: -0.8 */
  itkSetMacro(SigmoidMin, double);
  itkGetConstMacro(SigmoidMin, double);

  /** Set/Get the scaling of the sigmoid width. Large values
   * cause a more wide sigmoid. Default: 1e-8. Should be >0. */
  itkSetMacro(SigmoidScale, double);
  itkGetConstMacro(SigmoidScale, double);

protected:
  AdaptiveStochasticVarianceReducedGradientOptimizer();
  ~AdaptiveStochasticVarianceReducedGradientOptimizer() override = default;

  /** Function to update the current time
   * If UseAdaptiveStepSizes is false this function just increments
   * the CurrentTime by \f$E_0 = (sigmoid_{max} + sigmoid_{min})/2\f$.
   * Else, the CurrentTime is updated according to:\n
   * time = max[ 0, time + sigmoid( -gradient*previousgradient) ]\n
   * In that case, also the m_PreviousGradient is updated.
   */
  void
  UpdateCurrentTime() override;

  /** The PreviousGradient, necessary for the CruzAcceleration */
  DerivativeType m_PreviousGradient;

private:
  /** Settings */
  bool   m_UseAdaptiveStepSizes{ true };
  double m_SigmoidMax{ 1.0 };
  double m_SigmoidMin{ -0.8 };
  double m_SigmoidScale{ 1e-8 };

}; // end class AdaptiveStochasticVarianceReducedGradientOptimizer


} // end namespace itk


#endif // end #ifndef itkAdaptiveStochasticVarianceReducedGradientOptimizer_h
