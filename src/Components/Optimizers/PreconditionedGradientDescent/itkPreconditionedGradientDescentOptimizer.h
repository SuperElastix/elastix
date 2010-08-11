/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __itkPreconditionedGradientDescentOptimizer_h
#define __itkPreconditionedGradientDescentOptimizer_h

#include "itkScaledSingleValuedNonLinearOptimizer.h"
#include "vnl/algo/vnl_symmetric_eigensystem.h"
#include "itkArray2D.h"

namespace itk
{

  /** \class PreconditionedGradientDescentOptimizer
  * \brief Implement a gradient descent optimizer
  *
  * PreconditionedGradientDescentOptimizer implements a simple gradient descent optimizer.
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
  class PreconditionedGradientDescentOptimizer :
    public ScaledSingleValuedNonLinearOptimizer
  {
  public:
    /** Standard class typedefs. */
    typedef PreconditionedGradientDescentOptimizer               Self;
    typedef ScaledSingleValuedNonLinearOptimizer    Superclass;
    typedef SmartPointer<Self>                Pointer;
    typedef SmartPointer<const Self>          ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

    /** Run-time type information (and related methods). */
    itkTypeMacro( PreconditionedGradientDescentOptimizer, ScaledSingleValuedNonLinearOptimizer );

    /** Typedefs inherited from the superclass. */
    typedef Superclass::MeasureType               MeasureType;
    typedef Superclass::ParametersType            ParametersType;
    typedef Superclass::DerivativeType            DerivativeType;
    typedef Superclass::CostFunctionType          CostFunctionType;
    typedef Superclass::ScalesType                ScalesType;
    typedef Superclass::ScaledCostFunctionType    ScaledCostFunctionType;
    typedef Superclass::ScaledCostFunctionPointer ScaledCostFunctionPointer;

    /** Some typedefs for computing the SelfHessian */
    typedef DerivativeType::ValueType                       PreconditionValueType;
    typedef Array2D<PreconditionValueType>                  PreconditionType;
    typedef vnl_symmetric_eigensystem<
      PreconditionValueType >                               EigenSystemType;

    /** Codes of stopping conditions
     * The MinimumStepSize stopcondition never occurs, but may
     * be implemented in inheriting classes */
    typedef enum {
      MaximumNumberOfIterations,
      MetricError,
      MinimumStepSize } StopConditionType;

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
    itkGetConstReferenceMacro( LearningRate, double);

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

    /** Set the preconditioning matrix, whose inverse actually will be used to precondition.
     * On setting the precondition matrix, an eigensystem is computed immediately, the
     * eigenvalues/vectors are modified and only the modified eigenvectors/values are stored
     * (in the EigenSystem) */
    virtual void SetPreconditionMatrix( const PreconditionType & precondition );

    const EigenSystemType * GetEigenSystem(void) const
    {
      return this->m_EigenSystem;
    }

    /** Set the minimum condition number allowed for the PreconditionMatrix. Default: 0.01
     * In the SetPreconditionMatrix, the supplied precondition matrix will be modified such
     * that the condition number >= minimum condition number, after reducing the rank. */
    itkSetMacro( MinimumConditionNumber, double );
    itkGetConstMacro( MinimumConditionNumber, double );

  protected:
    PreconditionedGradientDescentOptimizer();
    virtual ~PreconditionedGradientDescentOptimizer();
    void PrintSelf(std::ostream& os, Indent indent) const;

    // made protected so subclass can access
    DerivativeType                m_Gradient;
    double                        m_LearningRate;
    StopConditionType             m_StopCondition;
    EigenSystemType *             m_EigenSystem;

  private:
    PreconditionedGradientDescentOptimizer(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented

    bool                          m_Stop;
    double                        m_Value;

    unsigned long                 m_NumberOfIterations;
    unsigned long                 m_CurrentIteration;

    double                        m_MinimumConditionNumber;

  };

} // end namespace itk


#endif



