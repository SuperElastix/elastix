/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkBSplineInterpolationDerivativeWeightFunction_hxx
#define __itkBSplineInterpolationDerivativeWeightFunction_hxx

#include "itkBSplineInterpolationDerivativeWeightFunction.h"

namespace itk
{

/**
 * ****************** Constructor *******************************
 */

template<class TCoordRep, unsigned int VSpaceDimension, unsigned int VSplineOrder>
BSplineInterpolationDerivativeWeightFunction<TCoordRep, VSpaceDimension, VSplineOrder>
::BSplineInterpolationDerivativeWeightFunction()
{
  /** Initialize members. */
  this->m_DerivativeDirection = 0;

  /** Initialize the interpolation kernels. */
  this->m_Kernel = KernelType::New();
  this->m_DerivativeKernel = DerivativeKernelType::New();

} // end Constructor


/**
 * ******************* SetDerivativeDirection *******************
 */

template<class TCoordRep, unsigned int VSpaceDimension, unsigned int VSplineOrder>
void
BSplineInterpolationDerivativeWeightFunction<TCoordRep, VSpaceDimension, VSplineOrder>
::SetDerivativeDirection( unsigned int dir )
{
  if ( dir != this->m_DerivativeDirection )
  {
    if ( dir < SpaceDimension )
    {
      this->m_DerivativeDirection = dir;

      this->Modified();
    }
  }

} // end SetDerivativeDirection()


/**
 * ******************* PrintSelf *******************
 */

template<class TCoordRep, unsigned int VSpaceDimension, unsigned int VSplineOrder>
void
BSplineInterpolationDerivativeWeightFunction<TCoordRep, VSpaceDimension, VSplineOrder>
::PrintSelf( std::ostream & os, Indent indent ) const
{
  this->Superclass::PrintSelf( os, indent );

  os << indent << "DerivativeDirection: "
    << this->m_DerivativeDirection << std::endl;

} // end PrintSelf()


/**
 * ******************* Compute1DWeights *******************
 */

template<class TCoordRep, unsigned int VSpaceDimension, unsigned int VSplineOrder>
void
BSplineInterpolationDerivativeWeightFunction<TCoordRep, VSpaceDimension, VSplineOrder>
::Compute1DWeights(
  const ContinuousIndexType & cindex,
  const IndexType & startIndex,
  OneDWeightsType & weights1D ) const
{
  /** Compute the 1D weights. */
  for ( unsigned int i = 0; i < SpaceDimension; ++i )
  {
    double x = cindex[ i ] - static_cast<double>( startIndex[ i ] );

    if ( i != this->m_DerivativeDirection )
    {
      for ( unsigned int k = 0; k < this->m_SupportSize[ i ]; ++k )
      {
        weights1D[ i ][ k ] = this->m_Kernel->Evaluate( x );
        x -= 1.0;
      }
    }
    else
    {
      for ( unsigned int k = 0; k < this->m_SupportSize[ i ]; ++k )
      {
        weights1D[ i ][ k ] = this->m_DerivativeKernel->Evaluate( x );
        x -= 1.0;
      }
    }
  }

} // end Compute1DWeights()


} // end namespace itk

#endif
