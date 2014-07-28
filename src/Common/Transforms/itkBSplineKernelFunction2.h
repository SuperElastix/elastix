/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkBSplineKernelFunction.h,v $
  Language:  C++
  Date:      $Date: 2006-03-18 20:13:35 $
  Version:   $Revision: 1.7 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkBSplineKernelFunction2_h
#define __itkBSplineKernelFunction2_h

#include "itkKernelFunctionBase.h"
#include "vnl/vnl_math.h"

namespace itk
{

/** \class BSplineKernelFunction2
 * \brief B-spline kernel used for density estimation and nonparameteric
 *  regression.
 *
 * This class encapsulates B-spline kernel for
 * density estimation or nonparametric regression.
 * See documentation for KernelFunction for more details.
 *
 * This class is templated over the spline order.
 * \warning Evaluate is only implemented for spline order 0 to 3
 *
 * \sa KernelFunction
 *
 * \ingroup Functions
 */
template< unsigned int VSplineOrder = 3 >
class ITK_EXPORT BSplineKernelFunction2 : public KernelFunctionBase< double >
{
public:

  /** Standard class typedefs. */
  typedef BSplineKernelFunction2       Self;
  typedef KernelFunctionBase< double > Superclass;
  typedef SmartPointer< Self >         Pointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( BSplineKernelFunction2, KernelFunctionBase );

  /** Enum of for spline order. */
  itkStaticConstMacro( SplineOrder, unsigned int, VSplineOrder );

  /** Store weights for the entire support. */
  typedef FixedArray< double,
    itkGetStaticConstMacro( SplineOrder ) + 1 >  WeightArrayType;

  /** Evaluate the function at one point. */
  inline double Evaluate( const double & u ) const
  {
    return this->Evaluate( Dispatch< VSplineOrder >(), u );
  }


  /** Evaluate the function at the entire support. This is slightly faster,
   * since no if's are needed.
   */
  inline void Evaluate( const double & u, WeightArrayType & weights ) const
  {
    this->Evaluate( Dispatch< VSplineOrder >(), u, weights );
  }


protected:

  BSplineKernelFunction2(){}
  ~BSplineKernelFunction2(){}

  void PrintSelf( std::ostream & os, Indent indent ) const
  {
    Superclass::PrintSelf( os, indent );
    os << indent << "Spline Order: " << SplineOrder << std::endl;
  }


private:

  BSplineKernelFunction2( const Self & ); //purposely not implemented
  void operator=( const Self & );         //purposely not implemented

  /** Structures to control overloaded versions of Evaluate */
  struct DispatchBase {};
  template< unsigned int >
  struct Dispatch : DispatchBase {};

  /** *****************************************************
   * B-spline functions for one point.
   */

  /** Zeroth order spline. */
  inline double Evaluate( const Dispatch< 0 > &, const double & u ) const
  {
    double absValue = vnl_math_abs( u );

    if( absValue < 0.5 ) { return 1.0; }
    else if( absValue == 0.5 ) { return 0.5; }
    else { return 0.0; }
  }


  /** First order spline */
  inline double Evaluate( const Dispatch< 1 > &, const double & u ) const
  {
    double absValue = vnl_math_abs( u );

    if( absValue < 1.0 ) { return 1.0 - absValue; }
    else { return 0.0; }
  }


  /** Second order spline. */
  inline double Evaluate( const Dispatch< 2 > &, const double & u ) const
  {
    double absValue = vnl_math_abs( u );

    if( absValue < 0.5 )
    {
      return 0.75 - vnl_math_sqr( absValue );
    }
    else if( absValue < 1.5 )
    {
      return ( 9.0 - 12.0 * absValue + 4.0 * vnl_math_sqr( absValue ) ) / 8.0;
    }
    else { return 0.0; }
  }


  /** Third order spline. */
  inline double Evaluate( const Dispatch< 3 > &, const double & u ) const
  {
    double absValue = vnl_math_abs( u );
    double sqrValue = vnl_math_sqr( u );

    if( absValue < 1.0 )
    {
      return ( 4.0 - 6.0 * sqrValue + 3.0 * sqrValue * absValue ) / 6.0;
    }
    else if( absValue < 2.0 )
    {
      return ( 8.0 - 12.0 * absValue + 6.0 * sqrValue - sqrValue * absValue ) / 6.0;
    }
    else { return 0.0; }
  }


  /** Unimplemented spline order. */
  inline double Evaluate( const DispatchBase &, const double & ) const
  {
    itkExceptionMacro( << "Evaluate not implemented for spline order "
                       << SplineOrder );
    return 0.0;
  }


  /** *****************************************************
   * B-spline functions for all points in the support.
   */

  /** Zeroth order spline. */
  inline void Evaluate( const Dispatch< 0 > &, const double & u,
    WeightArrayType & weights ) const
  {
    if( u < 0.5 ) { weights[ 0 ] = 1.0; }
    else { weights[ 0 ] = 0.5; }
  }


  /** First order spline */
  inline void Evaluate( const Dispatch< 1 > &, const double & u,
    WeightArrayType & weights ) const
  {
    weights[ 0 ] = 1.0 - u;
    weights[ 1 ] = u;
  }


  /** Second order spline. */
  inline void Evaluate( const Dispatch< 2 > &, const double & u,
    WeightArrayType & weights ) const
  {
    const double uu = vnl_math_sqr( u );

    weights[ 0 ] = ( 9.0 - 12.0 * u + 4.0 * uu ) / 8.0;
    weights[ 1 ] = -0.25 + 2.0 * u - uu;
    weights[ 2 ] = ( 1.0 - 4.0 * u + 4.0 * uu ) / 8.0;
  }


  /**  Third order spline. */
  inline void Evaluate( const Dispatch< 3 > &, const double & u,
    WeightArrayType & weights ) const
  {
    const double uu  = vnl_math_sqr( u );
    const double uuu = uu * u;

    weights[ 0 ] = ( 8.0 - 12 * u + 6.0 * uu - uuu ) / 6.0;
    weights[ 1 ] = ( -5.0 + 21.0 * u - 15.0 * uu + 3.0 * uuu ) / 6.0;
    weights[ 2 ] = ( 4.0 - 12.0 * u + 12.0 * uu - 3.0 * uuu ) / 6.0;
    weights[ 3 ] = ( -1.0 + 3.0 * u - 3.0 * uu + uuu ) / 6.0;
  }


  /** Unimplemented spline order. */
  inline double Evaluate( const DispatchBase &, const double &,
    WeightArrayType & ) const
  {
    itkExceptionMacro( << "Evaluate not implemented for spline order "
                       << SplineOrder );
    return 0.0;
  }


};

} // end namespace itk

#endif
