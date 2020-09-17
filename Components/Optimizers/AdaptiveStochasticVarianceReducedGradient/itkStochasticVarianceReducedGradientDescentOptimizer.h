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
#ifndef __itkStochasticVarianceReducedGradientDescentOptimizer_h
#define __itkStochasticVarianceReducedGradientDescentOptimizer_h

#include "itkScaledSingleValuedNonLinearOptimizer.h"
#include "itkPlatformMultiThreader.h"

namespace itk
{
/** \class StochasticVarianceReducedGradientDescentOptimizer
 * \brief Implement a gradient descent optimizer
 *
 * StochasticVarianceReducedGradientDescentOptimizer implements a simple gradient descent optimizer.
 * At each iteration the current position is updated according to
 *
 * \f[
 *        p_{n+1} = p_n
 *                + \mbox{learningRate} \, \frac{\partial f(p_n) }{\partial p_n}
 * \f]
 *
 * The learning rate is a fixed scalar defined via SetLearningRate().
 * The optimizer steps through a user defined number of iterations;
 * no convergence checking is done.
 *
 * Additionally, user can scale each component of the \f$\partial f / \partial p\f$
 * but setting a scaling vector using method SetScale().
 *
 * The difference of this class with the itk::GradientDescentOptimizer
 * is that it's based on the ScaledSingleValuedNonLinearOptimizer
 *
 * \sa ScaledSingleValuedNonLinearOptimizer
 *
 * \ingroup Numerics Optimizers
 */

class StochasticVarianceReducedGradientDescentOptimizer :
  public ScaledSingleValuedNonLinearOptimizer
{
public:
  /** Standard class typedefs. */
  typedef StochasticVarianceReducedGradientDescentOptimizer               Self;
  typedef ScaledSingleValuedNonLinearOptimizer    Superclass;
  typedef SmartPointer<Self>                Pointer;
  typedef SmartPointer<const Self>          ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( StochasticVarianceReducedGradientDescentOptimizer, ScaledSingleValuedNonLinearOptimizer );

  /** Typedefs inherited from the superclass. */
  typedef Superclass::MeasureType               MeasureType;
  typedef Superclass::ParametersType            ParametersType;
  typedef Superclass::DerivativeType            DerivativeType;
  typedef Superclass::CostFunctionType          CostFunctionType;
  typedef Superclass::ScalesType                ScalesType;
  typedef Superclass::ScaledCostFunctionType    ScaledCostFunctionType;
  typedef Superclass::ScaledCostFunctionPointer ScaledCostFunctionPointer;

  /** Codes of stopping conditions
   * The MinimumStepSize stop condition never occurs, but may
   * be implemented in inheriting classes */
  typedef enum {
    MaximumNumberOfIterations,
    MetricError,
    MinimumStepSize,
    InvalidDiagonalMatrix,
    GradientMagnitudeTolerance,
    LineSearchError
  } StopConditionType;

  /** Advance one step following the gradient direction. */
  virtual void AdvanceOneStep( void );

  /** Start optimization. */
  void StartOptimization( void ) override;

  /** Resume previously stopped optimization with current parameters
   * \sa StopOptimization. */
  virtual void ResumeOptimization( void );

  /** Stop optimization and pass on exception. */
  virtual void MetricErrorResponse( ExceptionObject & err );

  /** Stop optimization.
   * \sa ResumeOptimization */
  virtual void StopOptimization( void );

  /** Set the learning rate. */
  itkSetMacro( LearningRate, double );

  /** Get the learning rate. */
  itkGetConstReferenceMacro( LearningRate, double );

  /** Set the number of iterations. */
  itkSetMacro( NumberOfIterations, unsigned long );

  /** Get the inner LBFGSMemory. */
  itkGetConstMacro( LBFGSMemory, unsigned int );

  /** Get the number of iterations. */
  itkGetConstReferenceMacro( NumberOfIterations, unsigned long );

  /** Get the number of inner loop iterations. */
  itkGetConstReferenceMacro( NumberOfInnerIterations, unsigned long );

  /** Get the current iteration number. */
  itkGetConstMacro( CurrentIteration, unsigned int );

  /** Get the current inner iteration number. */
  itkGetConstMacro( CurrentInnerIteration, unsigned int );

  /** Get the current value. */
  itkGetConstReferenceMacro( Value, double );

  /** Get Stop condition. */
  itkGetConstReferenceMacro( StopCondition, StopConditionType );

  /** Get current gradient. */
  itkGetConstReferenceMacro( Gradient, DerivativeType );

  /** Get current search direction. */
  itkGetConstReferenceMacro( SearchDir, DerivativeType );

  /** Set the Previous Position. */
  itkSetMacro( PreviousPosition, ParametersType );

  /** Get the Previous Position. */
  itkGetConstReferenceMacro( PreviousPosition, ParametersType );

  /** Set the Previous gradient. */
  itkSetMacro( PreviousGradient, DerivativeType );

  /** Get the Previous gradient. */
  itkGetConstReferenceMacro( PreviousGradient, DerivativeType );

  /** Set the number of threads. */
  void SetNumberOfWorkUnits( ThreadIdType numberOfThreads )
  {
    this->m_Threader->SetNumberOfWorkUnits( numberOfThreads );
  }
  //itkGetConstReferenceMacro( NumberOfThreads, ThreadIdType );
  itkSetMacro( UseMultiThread, bool );

  itkSetMacro( UseOpenMP, bool );
  itkSetMacro( UseEigen, bool );

protected:
  StochasticVarianceReducedGradientDescentOptimizer();
  ~StochasticVarianceReducedGradientDescentOptimizer() override {};
  void PrintSelf( std::ostream& os, Indent indent ) const override;

  /** Typedefs for multi-threading. */
  typedef itk::PlatformMultiThreader               ThreaderType;
  typedef ThreaderType::WorkUnitInfo   ThreadInfoType;

  // made protected so subclass can access
  double                        m_Value;
  DerivativeType                m_Gradient;
  ParametersType                m_SearchDir;
  ParametersType                m_PreviousSearchDir;
  //ParametersType                m_PrePreviousSearchDir;
  ParametersType                m_MeanSearchDir;
  double                        m_LearningRate;
  StopConditionType             m_StopCondition;
  DerivativeType                m_PreviousGradient;
  //DerivativeType                m_PrePreviousGradient;
  ParametersType                m_PreviousPosition;
  ThreaderType::Pointer         m_Threader;

  bool                          m_Stop;
  unsigned long                 m_NumberOfIterations;
  unsigned long                 m_NumberOfInnerIterations;
  unsigned long                 m_CurrentIteration;
  unsigned long                 m_CurrentInnerIteration;
  unsigned long                 m_LBFGSMemory;

private:
  StochasticVarianceReducedGradientDescentOptimizer( const Self& ); // purposely not implemented
  void operator=( const Self& ); // purposely not implemented

  // multi-threaded AdvanceOneStep:
  bool m_UseMultiThread;
  struct MultiThreaderParameterType
  {
    ParametersType *  t_NewPosition;
    Self *            t_Optimizer;
  };

  bool m_UseOpenMP;
  bool m_UseEigen;

  /** The callback function. */
  static ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION AdvanceOneStepThreaderCallback( void * arg );

  /** The threaded implementation of AdvanceOneStep(). */
  inline void ThreadedAdvanceOneStep( ThreadIdType threadId, ParametersType & newPosition );

};

} // end namespace itk


#endif
