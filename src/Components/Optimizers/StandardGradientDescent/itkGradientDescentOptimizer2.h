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

#ifndef __itkGradientDescentOptimizer2_h
#define __itkGradientDescentOptimizer2_h

#include "itkScaledSingleValuedNonLinearOptimizer.h"
#include "itkMultiThreader.h"

namespace itk
{

/** \class GradientDescentOptimizer2
* \brief Implement a gradient descent optimizer
*
* GradientDescentOptimizer2 implements a simple gradient descent optimizer.
* At each iteration the current position is updated according to
*
* \f[
*        p_{n+1} = p_n
*                + \mbox{learningRate}
\, \frac{\partial f(p_n) }{\partial p_n}
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
class GradientDescentOptimizer2 :
  public ScaledSingleValuedNonLinearOptimizer
{
public:

  /** Standard class typedefs. */
  typedef GradientDescentOptimizer2            Self;
  typedef ScaledSingleValuedNonLinearOptimizer Superclass;
  typedef SmartPointer< Self >                 Pointer;
  typedef SmartPointer< const Self >           ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GradientDescentOptimizer2, ScaledSingleValuedNonLinearOptimizer );

  /** Typedefs inherited from the superclass. */
  typedef Superclass::MeasureType               MeasureType;
  typedef Superclass::ParametersType            ParametersType;
  typedef Superclass::DerivativeType            DerivativeType;
  typedef Superclass::CostFunctionType          CostFunctionType;
  typedef Superclass::ScalesType                ScalesType;
  typedef Superclass::ScaledCostFunctionType    ScaledCostFunctionType;
  typedef Superclass::ScaledCostFunctionPointer ScaledCostFunctionPointer;

  /** Codes of stopping conditions
   * The MinimumStepSize stopcondition never occurs, but may
   * be implemented in inheriting classes */
  typedef enum {
    MaximumNumberOfIterations,
    MetricError,
    MinimumStepSize
  } StopConditionType;

  /** Advance one step following the gradient direction. */
  virtual void AdvanceOneStep( void );

  /** Start optimization. */
  virtual void StartOptimization( void );

  /** Resume previously stopped optimization with current parameters
  * \sa StopOptimization. */
  virtual void ResumeOptimization( void );

  /** Stop optimisation and pass on exception. */
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

  /** Get the number of iterations. */
  itkGetConstReferenceMacro( NumberOfIterations, unsigned long );

  /** Get the current iteration number. */
  itkGetConstMacro( CurrentIteration, unsigned int );

  /** Get the current value. */
  itkGetConstReferenceMacro( Value, double );

  /** Get Stop condition. */
  itkGetConstReferenceMacro( StopCondition, StopConditionType );

  /** Get current gradient. */
  itkGetConstReferenceMacro( Gradient, DerivativeType );

  /** Set the number of threads. */
  void SetNumberOfThreads( ThreadIdType numberOfThreads )
  {
    this->m_Threader->SetNumberOfThreads( numberOfThreads );
  }


  //itkGetConstReferenceMacro( NumberOfThreads, ThreadIdType );
  itkSetMacro( UseMultiThread, bool );

  itkSetMacro( UseOpenMP, bool );
  itkSetMacro( UseEigen, bool );

protected:

  GradientDescentOptimizer2();
  virtual ~GradientDescentOptimizer2() {}
  void PrintSelf( std::ostream & os, Indent indent ) const;

  /** Typedefs for multi-threading. */
  typedef itk::MultiThreader             ThreaderType;
  typedef ThreaderType::ThreadInfoStruct ThreadInfoType;

  // made protected so subclass can access
  double            m_Value;
  DerivativeType    m_Gradient;
  double            m_LearningRate;
  StopConditionType m_StopCondition;

  ThreaderType::Pointer m_Threader;

  bool          m_Stop;
  unsigned long m_NumberOfIterations;
  unsigned long m_CurrentIteration;

private:

  GradientDescentOptimizer2( const Self & ); // purposely not implemented
  void operator=( const Self & );            // purposely not implemented

  // multi-threaded AdvanceOneStep:
  bool m_UseMultiThread;
  struct MultiThreaderParameterType
  {
    ParametersType * t_NewPosition;
    Self *           t_Optimizer;
  };

  bool m_UseOpenMP;
  bool m_UseEigen;

  /** The callback function. */
  static ITK_THREAD_RETURN_TYPE AdvanceOneStepThreaderCallback( void * arg );

  /** The threaded implementation of AdvanceOneStep(). */
  inline void ThreadedAdvanceOneStep( ThreadIdType threadId, ParametersType & newPosition );

};

} // end namespace itk

#endif
