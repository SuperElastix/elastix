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

/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkReducedDimensionLinearInterpolateImageFunction.txx,v $
  Language:  C++
  Date:      $Date: 2008-11-10 16:55:00 $
  Version:   $Revision: 1.21 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

  Portions of this code are covered under the VTK copyright.
  See VTKCopyright.txt or http://www.kitware.com/VTKCopyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkReducedDimensionLinearInterpolateImageFunction_hxx
#define __itkReducedDimensionLinearInterpolateImageFunction_hxx

#include "itkReducedDimensionLinearInterpolateImageFunction.h"

#include "vnl/vnl_math.h"

namespace itk
{

/**
 * Constructor
 */
template< class TImageType, class TCoordRep>
ReducedDimensionLinearInterpolateImageFunction< TImageType, TCoordRep>
::ReducedDimensionLinearInterpolateImageFunction()
{}


/**
 * Standard "PrintSelf" method
 */
template< class TImageType, class TCoordRep>
void
ReducedDimensionLinearInterpolateImageFunction< TImageType, TCoordRep>
::PrintSelf(
  std::ostream & os,
  Indent indent ) const
{
}
    template< class TImageType, class TCoordRep>
    typename ReducedDimensionLinearInterpolateImageFunction< TImageType, TCoordRep>
    ::OutputType
    ReducedDimensionLinearInterpolateImageFunction< TImageType, TCoordRep>
    ::EvaluateOptimized(const Dispatch< 3 > &, const ContinuousIndexType & index) const
    {
        IndexType basei;
        
        basei[0] = Math::Floor< IndexValueType >(index[0]);
        if ( basei[0] < this->m_StartIndex[0] )
        {
            basei[0] = this->m_StartIndex[0];
        }
        const ContinuousIndexValueType & distance0 = index[0] - static_cast< ContinuousIndexValueType >( basei[0] );
        
        basei[1] = Math::Floor< IndexValueType >(index[1]);
        if ( basei[1] < this->m_StartIndex[1] )
        {
            basei[1] = this->m_StartIndex[1];
        }
        const ContinuousIndexValueType & distance1 = index[1] - static_cast< ContinuousIndexValueType >( basei[1] );
        
        const TImageType * const inputImagePtr = this->GetInputImage();
        const RealType & val00 = inputImagePtr->GetPixel(basei);
        if ( distance0 <= 0. && distance1 <= 0. )
        {
            return ( static_cast< OutputType >( val00 ) );
        }
        else if ( distance1 <= 0. ) // if they have the same "y"
        {
            ++basei[0];  // then interpolate across "x"
            if ( basei[0] > this->m_EndIndex[0] )
            {
                return ( static_cast< OutputType >( val00 ) );
            }
            const RealType & val10 = inputImagePtr->GetPixel(basei);
            return ( static_cast< OutputType >( val00 + ( val10 - val00 ) * distance0 ) );
        }
        else if ( distance0 <= 0. ) // if they have the same "x"
        {
            ++basei[1];  // then interpolate across "y"
            if ( basei[1] > this->m_EndIndex[1] )
            {
                return ( static_cast< OutputType >( val00 ) );
            }
            const RealType & val01 = inputImagePtr->GetPixel(basei);
            return ( static_cast< OutputType >( val00 + ( val01 - val00 ) * distance1 ) );
        }
        // fall-through case:
        // interpolate across "xy"
        ++basei[0];
        if ( basei[0] > this->m_EndIndex[0] ) // interpolate across "y"
        {
            --basei[0];
            ++basei[1];
            if ( basei[1] > this->m_EndIndex[1] )
            {
                return ( static_cast< OutputType >( val00 ) );
            }
            const RealType & val01 = inputImagePtr->GetPixel(basei);
            return ( static_cast< OutputType >( val00 + ( val01 - val00 ) * distance1 ) );
        }
        const RealType & val10 = inputImagePtr->GetPixel(basei);
        
        const RealType & valx0 = val00 + ( val10 - val00 ) * distance0;
        
        ++basei[1];
        if ( basei[1] > this->m_EndIndex[1] ) // interpolate across "x"
        {
            return ( static_cast< OutputType >( valx0 ) );
        }
        const RealType & val11 = inputImagePtr->GetPixel(basei);
        --basei[0];
        const RealType & val01 = inputImagePtr->GetPixel(basei);
        
        const RealType & valx1 = val01 + ( val11 - val01 ) * distance0;
        
        return ( static_cast< OutputType >( valx0 + ( valx1 - valx0 ) * distance1 ) );

    }
    
    template< class TImageType, class TCoordRep>
    typename ReducedDimensionLinearInterpolateImageFunction< TImageType, TCoordRep>
    ::OutputType
    ReducedDimensionLinearInterpolateImageFunction< TImageType, TCoordRep>
    ::EvaluateOptimized(const Dispatch< 4 > &, const ContinuousIndexType & index) const
    {
        IndexType basei;
        basei[0] = Math::Floor< IndexValueType >(index[0]);
        if ( basei[0] < this->m_StartIndex[0] )
        {
            basei[0] = this->m_StartIndex[0];
        }
        const ContinuousIndexValueType & distance0 = index[0] - static_cast< ContinuousIndexValueType >( basei[0] );
        
        basei[1] = Math::Floor< IndexValueType >(index[1]);
        if ( basei[1] < this->m_StartIndex[1] )
        {
            basei[1] = this->m_StartIndex[1];
        }
        const ContinuousIndexValueType & distance1 = index[1] - static_cast< ContinuousIndexValueType >( basei[1] );
        
        basei[2] = Math::Floor< IndexValueType >(index[2]);
        if ( basei[2] < this->m_StartIndex[2] )
        {
            basei[2] = this->m_StartIndex[2];
        }
        const ContinuousIndexValueType & distance2 = index[2] - static_cast< ContinuousIndexValueType >( basei[2] );
        
        const TImageType * const inputImagePtr = this->GetInputImage();
        const RealType & val000 = inputImagePtr->GetPixel(basei);
        if ( distance0 <= 0. && distance1 <= 0. && distance2 <= 0. )
        {
            return ( static_cast< OutputType >( val000 ) );
        }
        
        if ( distance2 <= 0. )
        {
            if ( distance1 <= 0. ) // interpolate across "x"
            {
                ++basei[0];
                if ( basei[0] > this->m_EndIndex[0] )
                {
                    return ( static_cast< OutputType >( val000 ) );
                }
                const RealType & val100 = inputImagePtr->GetPixel(basei);
                
                return static_cast< OutputType >( val000 + ( val100 - val000 ) * distance0 );
            }
            else if ( distance0 <= 0. ) // interpolate across "y"
            {
                ++basei[1];
                if ( basei[1] > this->m_EndIndex[1] )
                {
                    return ( static_cast< OutputType >( val000 ) );
                }
                const RealType & val010 = inputImagePtr->GetPixel(basei);
                
                return static_cast< OutputType >( val000 + ( val010 - val000 ) * distance1 );
            }
            else  // interpolate across "xy"
            {
                ++basei[0];
                if ( basei[0] > this->m_EndIndex[0] ) // interpolate across "y"
                {
                    --basei[0];
                    ++basei[1];
                    if ( basei[1] > this->m_EndIndex[1] )
                    {
                        return ( static_cast< OutputType >( val000 ) );
                    }
                    const RealType & val010 = inputImagePtr->GetPixel(basei);
                    return static_cast< OutputType >( val000 + ( val010 - val000 ) * distance1 );
                }
                const RealType & val100 = inputImagePtr->GetPixel(basei);
                const RealType & valx00 = val000 + ( val100 - val000 ) * distance0;
                
                ++basei[1];
                if ( basei[1] > this->m_EndIndex[1] ) // interpolate across "x"
                {
                    return ( static_cast< OutputType >( valx00 ) );
                }
                const RealType & val110 = inputImagePtr->GetPixel(basei);
                
                --basei[0];
                const RealType & val010 = inputImagePtr->GetPixel(basei);
                const RealType & valx10 = val010 + ( val110 - val010 ) * distance0;
                
                return static_cast< OutputType >( valx00 + ( valx10 - valx00 ) * distance1 );
            }
        }
        else
        {
            if ( distance1 <= 0. )
            {
                if ( distance0 <= 0. ) // interpolate across "z"
                {
                    ++basei[2];
                    if ( basei[2] > this->m_EndIndex[2] )
                    {
                        return ( static_cast< OutputType >( val000 ) );
                    }
                    const RealType & val001 = inputImagePtr->GetPixel(basei);
                    
                    return static_cast< OutputType >( val000 + ( val001 - val000 ) * distance2 );
                }
                else // interpolate across "xz"
                {
                    ++basei[0];
                    if ( basei[0] > this->m_EndIndex[0] ) // interpolate across "z"
                    {
                        --basei[0];
                        ++basei[2];
                        if ( basei[2] > this->m_EndIndex[2] )
                        {
                            return ( static_cast< OutputType >( val000 ) );
                        }
                        const RealType & val001 = inputImagePtr->GetPixel(basei);
                        
                        return static_cast< OutputType >( val000 + ( val001 - val000 ) * distance2 );
                    }
                    const RealType & val100 = inputImagePtr->GetPixel(basei);
                    
                    const RealType & valx00 = val000 + ( val100 - val000 ) * distance0;
                    
                    ++basei[2];
                    if ( basei[2] > this->m_EndIndex[2] ) // interpolate across "x"
                    {
                        return ( static_cast< OutputType >( valx00 ) );
                    }
                    const RealType & val101 = inputImagePtr->GetPixel(basei);
                    
                    --basei[0];
                    const RealType & val001 = inputImagePtr->GetPixel(basei);
                    
                    const RealType & valx01 = val001 + ( val101 - val001 ) * distance0;
                    
                    return static_cast< OutputType >( valx00 + ( valx01 - valx00 ) * distance2 );
                }
            }
            else if ( distance0 <= 0. ) // interpolate across "yz"
            {
                ++basei[1];
                if ( basei[1] > this->m_EndIndex[1] ) // interpolate across "z"
                {
                    --basei[1];
                    ++basei[2];
                    if ( basei[2] > this->m_EndIndex[2] )
                    {
                        return ( static_cast< OutputType >( val000 ) );
                    }
                    const RealType & val001 = inputImagePtr->GetPixel(basei);
                    
                    return static_cast< OutputType >( val000 + ( val001 - val000 ) * distance2 );
                }
                const RealType & val010 = inputImagePtr->GetPixel(basei);
                
                const RealType & val0x0 = val000 + ( val010 - val000 ) * distance1;
                
                ++basei[2];
                if ( basei[2] > this->m_EndIndex[2] ) // interpolate across "y"
                {
                    return ( static_cast< OutputType >( val0x0 ) );
                }
                const RealType & val011 = inputImagePtr->GetPixel(basei);
                
                --basei[1];
                const RealType & val001 = inputImagePtr->GetPixel(basei);
                
                const RealType & val0x1 = val001 + ( val011 - val001 ) * distance1;
                
                return static_cast< OutputType >( val0x0 + ( val0x1 - val0x0 ) * distance2 );
            }
            else // interpolate across "xyz"
            {
                ++basei[0];
                if ( basei[0] > this->m_EndIndex[0] ) // interpolate across "yz"
                {
                    --basei[0];
                    ++basei[1];
                    if ( basei[1] > this->m_EndIndex[1] )  // interpolate across "z"
                    {
                        --basei[1];
                        ++basei[2];
                        if ( basei[2] > this->m_EndIndex[2] )
                        {
                            return ( static_cast< OutputType >( val000 ) );
                        }
                        const RealType & val001 = inputImagePtr->GetPixel(basei);
                        
                        return static_cast< OutputType >( val000 + ( val001 - val000 ) * distance2 );
                    }
                    const RealType & val010 = inputImagePtr->GetPixel(basei);
                    const RealType & val0x0 = val000 + ( val010 - val000 ) * distance1;
                    
                    ++basei[2];
                    if ( basei[2] > this->m_EndIndex[2] ) // interpolate across "y"
                    {
                        return ( static_cast< OutputType >( val0x0 ) );
                    }
                    const RealType & val011 = inputImagePtr->GetPixel(basei);
                    
                    --basei[1];
                    const RealType & val001 = inputImagePtr->GetPixel(basei);
                    
                    const RealType & val0x1 = val001 + ( val011 - val001 ) * distance1;
                    
                    return static_cast< OutputType >( val0x0 + ( val0x1 - val0x0 ) * distance2 );
                }
                const RealType & val100 = inputImagePtr->GetPixel(basei);
                
                const RealType & valx00 = val000 + ( val100 - val000 ) * distance0;
                
                ++basei[1];
                if ( basei[1] > this->m_EndIndex[1] ) // interpolate across "xz"
                {
                    --basei[1];
                    ++basei[2];
                    if ( basei[2] > this->m_EndIndex[2] ) // interpolate across "x"
                    {
                        return ( static_cast< OutputType >( valx00 ) );
                    }
                    const RealType & val101 = inputImagePtr->GetPixel(basei);
                    
                    --basei[0];
                    const RealType & val001 = inputImagePtr->GetPixel(basei);
                    
                    const RealType & valx01 = val001 + ( val101 - val001 ) * distance0;
                    
                    return static_cast< OutputType >( valx00 + ( valx01 - valx00 ) * distance2 );
                }
                const RealType & val110 = inputImagePtr->GetPixel(basei);
                
                --basei[0];
                const RealType & val010 = inputImagePtr->GetPixel(basei);
                
                const RealType & valx10 = val010 + ( val110 - val010 ) * distance0;
                
                const RealType & valxx0 = valx00 + ( valx10 - valx00 ) * distance1;
                
                ++basei[2];
                if ( basei[2] > this->m_EndIndex[2] ) // interpolate across "xy"
                {
                    return ( static_cast< OutputType >( valxx0 ) );
                }
                const RealType & val011 = inputImagePtr->GetPixel(basei);
                
                ++basei[0];
                const RealType & val111 = inputImagePtr->GetPixel(basei);
                
                --basei[1];
                const RealType & val101 = inputImagePtr->GetPixel(basei);
                
                --basei[0];
                const RealType & val001 = inputImagePtr->GetPixel(basei);
                
                const RealType & valx01 = val001 + ( val101 - val001 ) * distance0;
                const RealType & valx11 = val011 + ( val111 - val011 ) * distance0;
                const RealType & valxx1 = valx01 + ( valx11 - valx01 ) * distance1;
                
                return ( static_cast< OutputType >( valxx0 + ( valxx1 - valxx0 ) * distance2 ) );
            }
        }
    }

/**
 * ***************** EvaluateValueAndDerivativeOptimized ***********************
 */
    
template< class TImageType, class TCoordRep>
void
ReducedDimensionLinearInterpolateImageFunction< TImageType, TCoordRep>
::EvaluateValueAndDerivativeOptimized(const Dispatch< 3 > &, const ContinuousIndexType & x, OutputType & value, CovariantVectorType & deriv ) const
{
    // Get some handles
    const InputImageType *        inputImage = this->GetInputImage();
    const InputImageSpacingType & spacing    = inputImage->GetSpacing();
    
    /** Create a possibly mirrored version of x. */
    ContinuousIndexType xm = x;
    double              deriv_sign[ ImageDimension -1 ];
    for( unsigned int dim = 0; dim < (ImageDimension -1); dim++ )
    {
        deriv_sign[ dim ] = 1.0 / spacing[ dim ];
        if( x[ dim ] < this->m_StartIndex[ dim ] )
        {
            xm[ dim ]          = 2.0 * this->m_StartIndex[ dim ] - x[ dim ];
            deriv_sign[ dim ] *= -1.0;
        }
        if( x[ dim ] > this->m_EndIndex[ dim ] )
        {
            xm[ dim ]          = 2.0 * this->m_EndIndex[ dim ] - x[ dim ];
            deriv_sign[ dim ] *= -1.0;
        }
            
        /** Separately deal with cases on the image edge. */
        if( Math::FloatAlmostEqual( xm[ dim ], static_cast<ContinuousIndexValueType>( this->m_EndIndex[ dim ] ) ) )
        {
            xm[ dim ] -= 0.000001;
        }
    }
    // if this is mirrored again outside the image domain, then too bad.
        
    /**
     * Compute base index = closest index below point
     * Compute distance from point to base index
     */
    IndexType baseIndex;
    double    dist[ ImageDimension - 1 ];
    double    dinv[ ImageDimension - 1 ];
    for( unsigned int dim = 0; dim < (ImageDimension-1); dim++ )
    {
        baseIndex[ dim ] = Math::Floor< IndexValueType >( xm[ dim ] );
        
        dist[ dim ] = xm[ dim ] - static_cast< double >( baseIndex[ dim ] );
        dinv[ dim ] = 1.0 - dist[ dim ];
    }
        
    /** Get the 4 corner values. */
    const RealType val00 = inputImage->GetPixel( baseIndex );
    ++baseIndex[ 0 ];
    const RealType val10 = inputImage->GetPixel( baseIndex );
    --baseIndex[ 0 ]; ++baseIndex[ 1 ];
    const RealType val01 = inputImage->GetPixel( baseIndex );
    ++baseIndex[ 0 ];
    const RealType val11 = inputImage->GetPixel( baseIndex );
    
    /** Interpolate to get the value. */
    value = static_cast< OutputType >(val00 * dinv[ 0 ] * dinv[ 1 ] + val10 * dist[ 0 ] * dinv[ 1 ] + val01 * dinv[ 0 ] * dist[ 1 ] + val11 * dist[ 0 ] * dist[ 1 ] );
        
    /** Interpolate to get the derivative. */
    deriv[ 0 ] = deriv_sign[ 0 ] * ( dinv[ 1 ] * ( val10 - val00 ) + dist[ 1 ] * ( val11 - val01 ) );
    deriv[ 1 ] = deriv_sign[ 1 ] * ( dinv[ 0 ] * ( val01 - val00 ) + dist[ 0 ] * ( val11 - val10 ) );
        
    /** Take direction cosines into account. */
    CovariantVectorType orientedDerivative;
    inputImage->TransformLocalVectorToPhysicalVector( deriv, orientedDerivative );
    deriv = orientedDerivative;
        
} // end EvaluateValueAndDerivativeOptimized()

    
/**
 * ***************** EvaluateValueAndDerivativeOptimized ***********************
 */
    
template< class TImageType, class TCoordRep >
void
ReducedDimensionLinearInterpolateImageFunction< TImageType, TCoordRep >
::EvaluateValueAndDerivativeOptimized(const Dispatch< 4 > &, const ContinuousIndexType & x, OutputType & value, CovariantVectorType & deriv ) const
{
    // Get some handles
    const InputImageType *        inputImage = this->GetInputImage();
    const InputImageSpacingType & spacing    = inputImage->GetSpacing();
        
    /** Create a possibly mirrored version of x. */
    ContinuousIndexType xm = x;
    double              deriv_sign[ ImageDimension - 1 ];
    for( unsigned int dim = 0; dim < (ImageDimension-1); dim++ )
    {
        deriv_sign[ dim ] = 1.0 / spacing[ dim ];
        if( x[ dim ] < this->m_StartIndex[ dim ] )
        {
            xm[ dim ]          = 2.0 * this->m_StartIndex[ dim ] - x[ dim ];
            deriv_sign[ dim ] *= -1.0;
        }
        if( x[ dim ] > this->m_EndIndex[ dim ] )
        {
            xm[ dim ]          = 2.0 * this->m_EndIndex[ dim ] - x[ dim ];
            deriv_sign[ dim ] *= -1.0;
        }
            
        /** Separately deal with cases on the image edge. */
        if( Math::FloatAlmostEqual( xm[ dim ], static_cast<ContinuousIndexValueType>( this->m_EndIndex[ dim ] ) ) )
        {
            xm[ dim ] -= 0.000001;
        }
    }
    // if this is mirrored again outside the image domain, then too bad.
        
    /**
     * Compute base index = closest index below point
     * Compute distance from point to base index
     */
    IndexType baseIndex;
    double    dist[ ImageDimension - 1];
    double    dinv[ ImageDimension - 1];
    for( unsigned int dim = 0; dim < ( ImageDimension - 1 ); dim++ )
    {
        baseIndex[ dim ] = Math::Floor< IndexValueType >( xm[ dim ] );
            
        dist[ dim ] = xm[ dim ] - static_cast< double >( baseIndex[ dim ] );
        dinv[ dim ] = 1.0 - dist[ dim ];
    }
        
    /** Get the 8 corner values. */
    const RealType val000 = inputImage->GetPixel( baseIndex );
    ++baseIndex[ 0 ];
    const RealType val100 = inputImage->GetPixel( baseIndex );
    ++baseIndex[ 1 ];
    const RealType val110 = inputImage->GetPixel( baseIndex );
    ++baseIndex[ 2 ];
    const RealType val111 = inputImage->GetPixel( baseIndex );
    --baseIndex[ 1 ];
    const RealType val101 = inputImage->GetPixel( baseIndex );
    --baseIndex[ 0 ];
    const RealType val001 = inputImage->GetPixel( baseIndex );
    ++baseIndex[ 1 ];
    const RealType val011 = inputImage->GetPixel( baseIndex );
    --baseIndex[ 2 ];
    const RealType val010 = inputImage->GetPixel( baseIndex );
        
    /** Interpolate to get the value. */
    value = static_cast< OutputType >( val000 * dinv[ 0 ] * dinv[ 1 ] * dinv[ 2 ] + val100 * dist[ 0 ] * dinv[ 1 ] * dinv[ 2 ] + val010 * dinv[ 0 ] * dist[ 1 ] * dinv[ 2 ] + val001 * dinv[ 0 ] * dinv[ 1 ] * dist[ 2 ] + val110 * dist[ 0 ] * dist[ 1 ] * dinv[ 2 ] + val011 * dinv[ 0 ] * dist[ 1 ] * dist[ 2 ] + val101 * dist[ 0 ] * dinv[ 1 ] * dist[ 2 ] + val111 * dist[ 0 ] * dist[ 1 ] * dist[ 2 ] );
        
    /** Interpolate to get the derivative. */
    deriv[ 0 ] = deriv_sign[ 0 ] * ( dinv[ 1 ] * dinv[ 2 ] * ( val100 - val000 )
           + dist[ 1 ] * dinv[ 2 ] * ( val110 - val010 )
           + dinv[ 1 ] * dist[ 2 ] * ( val101 - val001 )
           + dist[ 1 ] * dist[ 2 ] * ( val111 - val011 )
           );
    deriv[ 1 ] = deriv_sign[ 1 ] * ( dinv[ 0 ] * dinv[ 2 ] * ( val010 - val000 )
           + dist[ 0 ] * dinv[ 2 ] * ( val110 - val100 )
           + dinv[ 0 ] * dist[ 2 ] * ( val011 - val001 )
           + dist[ 0 ] * dist[ 2 ] * ( val111 - val101 )
           );
    deriv[ 2 ] = deriv_sign[ 2 ] * ( dinv[ 0 ] * dinv[ 1 ] * ( val001 - val000 )
           + dist[ 0 ] * dinv[ 1 ] * ( val101 - val100 )
           + dinv[ 0 ] * dist[ 1 ] * ( val011 - val010 )
           + dist[ 0 ] * dist[ 1 ] * ( val111 - val110 )
           );
        
    /** Take direction cosines into account. */
    CovariantVectorType orientedDerivative;
    inputImage->TransformLocalVectorToPhysicalVector( deriv, orientedDerivative );
    deriv = orientedDerivative;
        
} // end EvaluateValueAndDerivativeOptimized()
    
    
} // end namespace itk

#endif
