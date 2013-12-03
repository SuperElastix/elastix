/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __itkLineSearchOptimizer_h
#define __itkLineSearchOptimizer_h

#include "itkSingleValuedNonLinearOptimizer.h"

#include "itkIntTypes.h"//tmp

namespace itk
{

/**
 * \class LineSearchOptimizer
 *
 * \brief A base class for LineSearch optimizers.
 *
 * Scales are expected to be handled by the main optimizer.
 */

class LineSearchOptimizer : public SingleValuedNonLinearOptimizer
{
public:

  typedef LineSearchOptimizer                   Self;
  typedef SingleValuedNonLinearOptimizer        Superclass;
  typedef SmartPointer<Self>                    Pointer;
  typedef SmartPointer<const Self>              ConstPointer;

  //itkNewMacro(Self); because this is an abstract base class.
  itkTypeMacro( LineSearchOptimizer, SingleValuedNonLinearOptimizer );

  typedef Superclass::MeasureType               MeasureType;
  typedef Superclass::ParametersType            ParametersType;
  typedef Superclass::DerivativeType            DerivativeType;
  typedef Superclass::CostFunctionType          CostFunctionType;

  /** Set/Get the LineSearchDirection */
  virtual void SetLineSearchDirection( const ParametersType & arg )
  {
    this->m_LineSearchDirection = arg;
    this->Modified();
  }
  itkGetConstReferenceMacro( LineSearchDirection, ParametersType );

  /** Inheriting classes may override these methods if they need
   * value/derivative information of the cost function at the
   * initial position.
   *
   * NB: It is not guaranteed that these methods are called.
   * If a main optimizer by chance has this information, it
   * should call these methods, to avoid possible unnecessary
   * computations.
   */
  virtual void SetInitialDerivative(
    const DerivativeType & itkNotUsed( derivative ) ) {};
  virtual void SetInitialValue( MeasureType itkNotUsed( value ) ) {};

  /** These methods must be implemented by inheriting classes. It
   * depends on the specific line search algorithm if it already computed
   * the value/derivative at the current position (in this case it
   * can just copy the cached data). If it did not
   * compute the value/derivative, it should call the cost function
   * and evaluate the value/derivative at the current position.
   *
   * These methods allow the main optimization algorithm to reuse
   * data that the LineSearch algorithm already computed.
   */
  virtual void GetCurrentValueAndDerivative(
    MeasureType & value, DerivativeType & derivative ) const = 0;
  virtual void GetCurrentDerivative( DerivativeType & derivative ) const = 0;
  virtual MeasureType GetCurrentValue( void ) const = 0;

  /**
   * StepLength is a a scalar, defined as:
   * m_InitialPosition + StepLength * m_LineSearchDirection  =
   * m_CurrentPosition
   */
  itkGetConstMacro( CurrentStepLength, double );

  /** Settings: the maximum/minimum step length and the initial
   * estimate.
   * NOTE: Not all line search methods are guaranteed to
   * do something with this information.
   * However, if a certain optimizer (using a line search
   * optimizer) has any idea about the steplength it can
   * call these methods, 'in the hope' that the line search
   * optimizer does something sensible with it.
   */
  itkSetMacro( MinimumStepLength, double );
  itkGetConstMacro( MinimumStepLength, double );
  itkSetMacro( MaximumStepLength, double );
  itkGetConstMacro( MaximumStepLength, double );
  itkSetMacro( InitialStepLengthEstimate, double );
  itkGetConstMacro( InitialStepLengthEstimate, double );

protected:

  LineSearchOptimizer();
  virtual ~LineSearchOptimizer() {};
  void PrintSelf( std::ostream& os, Indent indent ) const;

  double        m_CurrentStepLength;

  /** Set the current step length AND the current position, where
   * the current position is computed as:
   * m_CurrentPosition =
   * m_InitialPosition + StepLength * m_LineSearchDirection
   */
  virtual void SetCurrentStepLength( double step );

  /** Computes the inner product of the argument and the line search direction. */
  double DirectionalDerivative( const DerivativeType & derivative ) const;

private:
  LineSearchOptimizer(const Self&); // purposely not implemented
  void operator=(const Self&); // purposely not implemented

  ParametersType      m_LineSearchDirection;

  double              m_MinimumStepLength;
  double              m_MaximumStepLength;
  double              m_InitialStepLengthEstimate;

};

} // end namespace itk

#endif // #ifndef __itkLineSearchOptimizer_h
