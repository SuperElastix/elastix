/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __itkExponentialLimiterFunction_h
#define __itkExponentialLimiterFunction_h

#include "itkLimiterFunctionBase.h"


namespace itk
{

/**
 * \class ExponentialLimiterFunction
 * \brief A soft limiter
 *
 * If the input value exceeds the upper/lower threshold the output is
 * diminished/increased, such that it never will exceed the UpperBound/LowerBound.
 * It does this in a smooth manner, with an exponential function.
 *
 * \f[ L(f(x)) = (T-B) e^{(f-T)/(T-B)} + B, \f]
 * where \f$B\f$ is the upper/lower bound and \f$T\f$ the upper/lower threshold
 *
 * \ingroup Functions
 * \sa LimiterFunctionBase, HardLimiterFunction
 *
 */
template < class TInput, unsigned int NDimension >
class ExponentialLimiterFunction :
  public LimiterFunctionBase<TInput, NDimension>
{
public:
  /** Standard class typedefs. */
  typedef ExponentialLimiterFunction                     Self;
  typedef LimiterFunctionBase<TInput, NDimension> Superclass;
  typedef SmartPointer<Self> Pointer;
  typedef SmartPointer<const Self> ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro(ExponentialLimiterFunction, LimiterFunctionBase);

  /** Define the ::New() function, for creation via the ObjectFactory */
  itkNewMacro(Self);

  /** Superclass' static consts */
  itkStaticConstMacro( Dimension, unsigned int, Superclass::Dimension );


  /** Superclass' typedefs */
  typedef typename Superclass::InputType            InputType;
  typedef typename Superclass::OutputType           OutputType;
  typedef typename Superclass::DerivativeValueType  DerivativeValueType;
  typedef typename Superclass::DerivativeType       DerivativeType;

  /** Limit the input value */
  virtual OutputType Evaluate( const InputType & input ) const;

  /** Limit the input value and change the input function derivative accordingly */
  virtual OutputType Evaluate( const InputType & input, DerivativeType & derivative) const;

  /** Initialize the limiter; calls the ComputeLimiterSettings() function */
  virtual void Initialize(void) throw (ExceptionObject);

protected:
  ExponentialLimiterFunction();
  ~ExponentialLimiterFunction(){};

  virtual void ComputeLimiterSettings(void);

  double m_UTminUB;
  double m_UTminUBinv;
  double m_LTminLB;
  double m_LTminLBinv;


private:
  ExponentialLimiterFunction(const Self& ); //purposely not implemented
  void operator=(const Self& ); //purposely not implemented

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkExponentialLimiterFunction.hxx"
#endif

#endif
