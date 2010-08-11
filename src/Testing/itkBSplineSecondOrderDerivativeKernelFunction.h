/*======================================================================

This file is part of the elastix software.

Copyright (c) University Medical Center Utrecht. All rights reserved.
See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkBSplineSecondOrderDerivativeKernelFunction_h
#define __itkBSplineSecondOrderDerivativeKernelFunction_h

#include "itkKernelFunction.h"
#include "itkBSplineKernelFunction.h"

namespace itk
{

/** \class BSplineSecondOrderDerivativeKernelFunction
 * \brief Derivative of a BSpline kernel used for density estimation and
 *  nonparameteric regression.
 *
 * This class encapsulates the derivative of a BSpline kernel for
 * density estimation or nonparameteric regression.
 * See documentation for KernelFunction for more details.
 *
 * This class is templated over the spline order.
 * \warning Evaluate is only implemented for spline order 1 to 4
 *
 * \sa KernelFunction
 *
 * \ingroup Functions
 */
template <unsigned int VSplineOrder = 3>
class ITK_EXPORT BSplineSecondOrderDerivativeKernelFunction : public KernelFunction
{
public:
  /** Standard class typedefs. */
  typedef BSplineSecondOrderDerivativeKernelFunction Self;
  typedef KernelFunction                  Superclass;
  typedef SmartPointer<Self>              Pointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(BSplineSecondOrderDerivativeKernelFunction, KernelFunction);

  /** Enum of for spline order. */
  itkStaticConstMacro(SplineOrder, unsigned int, VSplineOrder);

  /** Evaluate the function. */
  inline double Evaluate( const double & u ) const
    {
    return ( m_KernelFunction->Evaluate( u + 1.0 )
      - 2.0 * m_KernelFunction->Evaluate( u )
      + m_KernelFunction->Evaluate( u - 1.0 ) );
    }

protected:

  typedef BSplineKernelFunction<itkGetStaticConstMacro(SplineOrder) - 2>
      KernelType;

  BSplineSecondOrderDerivativeKernelFunction()
    {
    m_KernelFunction = KernelType::New();
    }

  ~BSplineSecondOrderDerivativeKernelFunction(){};
  void PrintSelf(std::ostream& os, Indent indent) const
    {
    Superclass::PrintSelf( os, indent );
    os << indent  << "Spline Order: " << SplineOrder << std::endl;
    }

private:
  BSplineSecondOrderDerivativeKernelFunction(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  typename KernelType::Pointer  m_KernelFunction;

};

} // end namespace itk

#endif
