/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkBSplineInterpolationSecondOrderDerivativeWeightFunction_txx
#define __itkBSplineInterpolationSecondOrderDerivativeWeightFunction_txx

#include "itkBSplineInterpolationSecondOrderDerivativeWeightFunction.h"
#include "itkImage.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "vnl/vnl_vector.h"


namespace itk
{


/**
 * ****************** Constructor *******************************
 */

template<class TCoordRep, unsigned int VSpaceDimension, unsigned int VSplineOrder>
BSplineInterpolationSecondOrderDerivativeWeightFunction<TCoordRep, VSpaceDimension, VSplineOrder>
::BSplineInterpolationSecondOrderDerivativeWeightFunction()
{
  /** Initialize members. */
  this->m_DerivativeDirections.fill( 0 );
  this->m_EqualDerivativeDirections = true;

  /** Initialize the interpolation kernels. */
  this->m_Kernel = KernelType::New();
  this->m_DerivativeKernel = DerivativeKernelType::New();
  this->m_SecondOrderDerivativeKernel = SecondOrderDerivativeKernelType::New();

} // end Constructor


/**
 * ******************* SetDerivativeDirections *******************
 */

template<class TCoordRep, unsigned int VSpaceDimension, unsigned int VSplineOrder>
void
BSplineInterpolationSecondOrderDerivativeWeightFunction<TCoordRep, VSpaceDimension, VSplineOrder>
::SetDerivativeDirections( unsigned int dir0, unsigned int dir1 )
{
  if ( dir0 != this->m_DerivativeDirections[ 0 ]
    || dir1 != this->m_DerivativeDirections[ 1 ] )
  {
    if ( dir0 < SpaceDimension && dir1 < SpaceDimension )
    {
      this->m_DerivativeDirections[ 0 ] = dir0;
      this->m_DerivativeDirections[ 1 ] = dir1;
      this->m_EqualDerivativeDirections = false;
      if ( dir0 == dir1 )
      {
        this->m_EqualDerivativeDirections = true;
      }

      /** Change variables. *
      this->InitializeSupport();
      this->InitializeOffsetToIndexTable();*/

      this->Modified();
    }
  }

} // end SetDerivativeDirections()


/**
 * ******************* PrintSelf *******************
 */

template<class TCoordRep, unsigned int VSpaceDimension, unsigned int VSplineOrder>
void
BSplineInterpolationSecondOrderDerivativeWeightFunction<TCoordRep, VSpaceDimension, VSplineOrder>
::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );
  
  os << indent << "DerivativeDirections: ["
    << this->m_DerivativeDirections[ 0 ] << ", "
    << this->m_DerivativeDirections[ 1 ] << "]"
    << std::endl;
  os << indent << "EqualDerivativeDirections: "
    << this->m_EqualDerivativeDirections << std::endl;

} // end PrintSelf()


/**
 * ******************* Compute1DWeights *******************
 */

template<class TCoordRep, unsigned int VSpaceDimension, unsigned int VSplineOrder>
void
BSplineInterpolationSecondOrderDerivativeWeightFunction<TCoordRep, VSpaceDimension, VSplineOrder>
::Compute1DWeights(
  const ContinuousIndexType & index,
  const IndexType & startIndex,
  std::vector< vnl_vector< double > > & weights1D ) const
{
  /** Compute the 1D weights. */
  weights1D.resize( SpaceDimension );
  for ( unsigned int i = 0; i < SpaceDimension; ++i )
  {
    double x = index[ i ] - static_cast<double>( startIndex[ i ] );
    
    weights1D[ i ].set_size( this->m_SupportSize[ i ] );
    for ( unsigned int k = 0; k < this->m_SupportSize[ i ]; ++k )
    {
      if ( i != this->m_DerivativeDirections[ 0 ]
        && i != this->m_DerivativeDirections[ 1 ] )
      {
        weights1D[ i ][ k ] = this->m_Kernel->Evaluate( x );
      }
      else
      {
        if ( this->m_EqualDerivativeDirections )
        {
          weights1D[ i ][ k ] = this->m_SecondOrderDerivativeKernel->Evaluate( x );
        }
        else
        {
          weights1D[ i ][ k ] = this->m_DerivativeKernel->Evaluate( x );
        }
      }
      x -= 1.0;
    }
  }

} // end Compute1DWeights()


} // end namespace itk

#endif
