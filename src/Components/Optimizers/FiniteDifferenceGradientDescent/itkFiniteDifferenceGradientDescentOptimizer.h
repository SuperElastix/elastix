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

#ifndef __itkFiniteDifferenceGradientDescentOptimizer_h
#define __itkFiniteDifferenceGradientDescentOptimizer_h

#include "itkScaledSingleValuedNonLinearOptimizer.h"

namespace itk
{

/**
 * \class FiniteDifferenceGradientDescentOptimizer
 * \brief An optimizer based on gradient descent ...
 *
 * If \f$C(x)\f$ is a costfunction that has to be minimised, the following iterative
 * algorithm is used to find the optimal parameters \a x:
 *
 * \f[ x(k+1)_j = x(k)_j - a(k) \left[ C(x(k)_j + c(k)) - C(x(k)_j - c(k)) \right] / 2c(k), \f]
 * for all parameters \f$j\f$.
 *
 * From this equation it is clear that it a gradient descent optimizer, using
 * a finite difference approximation of the gradient.
 *
 * The gain \f$a(k)\f$ at each iteration \f$k\f$ is defined by:
 *
 * \f[ a(k) =  a / (A + k + 1)^{\alpha}. \f]
 *
 * The perturbation size \f$c(k)\f$ at each iteration \f$k\f$ is defined by:
 *
 * \f[ c(k) =  c / (k + 1)^{\gamma}. \f]
 *
 * Note the similarities to the SimultaneousPerturbation optimizer and
 * the StandardGradientDescent optimizer.
 *
 * \ingroup Optimizers
 * \sa FiniteDifferenceGradientDescent
 */

class FiniteDifferenceGradientDescentOptimizer :
  public ScaledSingleValuedNonLinearOptimizer
{
public:

  /** Standard class typedefs. */
  typedef FiniteDifferenceGradientDescentOptimizer Self;
  typedef ScaledSingleValuedNonLinearOptimizer     Superclass;
  typedef SmartPointer< Self >                     Pointer;
  typedef SmartPointer< const Self >               ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( FiniteDifferenceGradientDescentOptimizer, ScaledSingleValuedNonLinearOptimizer );

  /** Codes of stopping conditions */
  typedef enum {
    MaximumNumberOfIterations,
    MetricError
  } StopConditionType;

  /** Advance one step following the gradient direction. */
  virtual void AdvanceOneStep( void );

  /** Start optimization. */
  void StartOptimization( void );

  /** Resume previously stopped optimization with current parameters
  * \sa StopOptimization. */
  void ResumeOptimization( void );

  /** Stop optimization.
  * \sa ResumeOptimization */
  void StopOptimization( void );

  /** Set the number of iterations. */
  itkSetMacro( NumberOfIterations, unsigned long );

  /** Get the number of iterations. */
  itkGetConstMacro( NumberOfIterations, unsigned long );

  /** Get the current iteration number. */
  itkGetConstMacro( CurrentIteration, unsigned long );

  /** Get the current value. */
  itkGetConstMacro( Value, double );

  /** Get Stop condition. */
  itkGetConstMacro( StopCondition, StopConditionType );

  /** Set/Get a. */
  itkSetMacro( Param_a, double );
  itkGetMacro( Param_a, double );

  /** Set/Get c. */
  itkSetMacro( Param_c, double );
  itkGetMacro( Param_c, double );

  /** Set/Get A. */
  itkSetMacro( Param_A, double );
  itkGetMacro( Param_A, double );

  /** Set/Get alpha. */
  itkSetMacro( Param_alpha, double );
  itkGetMacro( Param_alpha, double );

  /** Set/Get gamma. */
  itkSetMacro( Param_gamma, double );
  itkGetMacro( Param_gamma, double );

  itkGetConstMacro( ComputeCurrentValue, bool );
  itkSetMacro( ComputeCurrentValue, bool );
  itkBooleanMacro( ComputeCurrentValue );

  /** Get the CurrentStepLength, GradientMagnitude and LearningRate (a_k) */
  itkGetConstMacro( GradientMagnitude, double );
  itkGetConstMacro( LearningRate, double );

protected:

  FiniteDifferenceGradientDescentOptimizer();
  virtual ~FiniteDifferenceGradientDescentOptimizer() {}

  /** PrintSelf method.*/
  void PrintSelf( std::ostream & os, Indent indent ) const;

  // made protected so subclass can access
  DerivativeType m_Gradient;
  double         m_LearningRate;
  double         m_GradientMagnitude;

  /** Boolean that says if the current value of
  * the metric has to be computed. This is not
  * necessary for optimisation; just nice for
  * progress information.
  */
  bool m_ComputeCurrentValue;

  // Functions to compute the parameters at iteration k.
  virtual double Compute_a( unsigned long k ) const;

  virtual double Compute_c( unsigned long k ) const;

private:

  FiniteDifferenceGradientDescentOptimizer( const Self & ); // purposely not implemented
  void operator=( const Self & );                           // purposely not implemented

  /** Private member variables.*/
  bool              m_Stop;
  double            m_Value;
  StopConditionType m_StopCondition;
  unsigned long     m_NumberOfIterations;
  unsigned long     m_CurrentIteration;

  /**Parameters, as described by Spall.*/
  double m_Param_a;
  double m_Param_c;
  double m_Param_A;
  double m_Param_alpha;
  double m_Param_gamma;

};

} // end namespace itk

#endif // end #ifndef __itkFiniteDifferenceGradientDescentOptimizer_h
