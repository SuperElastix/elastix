/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __itkHardLimiterFunction_h
#define __itkHardLimiterFunction_h

#include "itkLimiterFunctionBase.h"


namespace itk
{

/**
 * \class HardLimiterFunction
 * \brief A hard limiter
 *
 * If the input value exceeds the upper/lower bound the output is
 * set to the upper/lower bound and the derivative is filled with zeros.
 *
 * \ingroup Functions
 * \sa LimiterFunctionBase, ExponentialLimiterFunction
 *
 */
template < class TInput, unsigned int NDimension >
class HardLimiterFunction :
  public LimiterFunctionBase<TInput, NDimension>
{
public:
  /** Standard class typedefs. */
  typedef HardLimiterFunction                     Self;
  typedef LimiterFunctionBase<TInput, NDimension> Superclass;
  typedef SmartPointer<Self> Pointer;
  typedef SmartPointer<const Self> ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro(HardLimiterFunction, LimiterFunctionBase);

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

protected:
  HardLimiterFunction(){};
  ~HardLimiterFunction(){};

private:
  HardLimiterFunction(const Self& ); //purposely not implemented
  void operator=(const Self& ); //purposely not implemented

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkHardLimiterFunction.hxx"
#endif

#endif
