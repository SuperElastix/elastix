/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef __itkRecursiveBSplineInterpolationWeightFunction_hxx
#define __itkRecursiveBSplineInterpolationWeightFunction_hxx

#include "itkRecursiveBSplineInterpolationWeightFunction.h"
#include "itkImage.h"
#include "itkMatrix.h"
#include "itkMath.h"
#include "itkImageRegionConstIteratorWithIndex.h"

namespace itk
{
/** Constructor */
template< typename TCoordRep, unsigned int VSpaceDimension,
          unsigned int VSplineOrder >
RecursiveBSplineInterpolationWeightFunction< TCoordRep, VSpaceDimension, VSplineOrder > 
::RecursiveBSplineInterpolationWeightFunction()  
{
    // Initialize support region is a hypercube of length SplineOrder + 1
    m_SupportSize.Fill(SplineOrder + 1);

    this->m_NumberOfWeights = 1;
    for( unsigned int i = 0; i < SpaceDimension; ++i )
    {
        this->m_NumberOfWeights *= this->m_SupportSize[ i ];
    }

    // Initialize offset to index lookup table
    typedef Image< char, SpaceDimension > CharImageType;
    typename CharImageType::Pointer tempImage = CharImageType::New();
    tempImage->SetRegions(m_SupportSize);
    tempImage->Allocate();
    tempImage->FillBuffer(0);


    // Initialize the interpolation kernel
    this->m_Kernel = KernelType::New();
    this->m_DerivativeKernel = DerivativeKernelType::New();

    /** Initialize members. */
    this->m_DerivativeDirection = 0;
}

/**
 * Standard "PrintSelf" method
 */
template< typename TCoordRep, unsigned int VSpaceDimension,
          unsigned int VSplineOrder >
void
RecursiveBSplineInterpolationWeightFunction< TCoordRep, VSpaceDimension, VSplineOrder >//ok 
::PrintSelf(
        std::ostream & os,
        Indent indent) const
{
    Superclass::PrintSelf(os, indent);

    os << indent << "NumberOfWeights: " << m_NumberOfWeights << std::endl;
    os << indent << "SupportSize: " << m_SupportSize << std::endl;
}

/** Compute weights for interpolation at continuous index position */
template< typename TCoordRep, unsigned int VSpaceDimension, unsigned int VSplineOrder >
typename RecursiveBSplineInterpolationWeightFunction< TCoordRep, VSpaceDimension, //ok 
VSplineOrder >
::WeightsType
RecursiveBSplineInterpolationWeightFunction< TCoordRep, VSpaceDimension, VSplineOrder > //ok 
::Evaluate(
        const ContinuousIndexType & index) const
{
    std::cout << "Evaluate" << std::endl;
    std::cout << "Nr of weights: "  << this->m_NumberOfWeights << std::endl;
    WeightsType weights(this->m_NumberOfWeights);
    IndexType   startIndex;

    this->Evaluate(index, weights, startIndex);

    return weights;
}

/** Compute weights for interpolation at continuous index position */
template< typename TCoordRep, unsigned int VSpaceDimension, unsigned int VSplineOrder >
void RecursiveBSplineInterpolationWeightFunction< TCoordRep, VSpaceDimension, VSplineOrder >
::Evaluate(
        const ContinuousIndexType & index,
        WeightsType & weights,
        IndexType & startIndex) const
{
    unsigned int j, k;

    // Find the starting index of the support region
    k=0;
    for ( j = 0; j < SpaceDimension; j++ )
    {
        startIndex[j] = Math::Floor< IndexValueType >(index[j] - static_cast< double >( SplineOrder - 1 ) / 2.0);
        double x = index[j] - static_cast< double >( startIndex[j] );
        for(unsigned int l = 0; l < SplineOrder+1; ++l)
        {
            weights[k] = m_Kernel->Evaluate(x);
            x -= 1.0;
            ++k;
        }
    }
} 

template< typename TCoordRep, unsigned int VSpaceDimension, unsigned int VSplineOrder >
void RecursiveBSplineInterpolationWeightFunction< TCoordRep, VSpaceDimension, VSplineOrder >
::EvaluateDerivative(
        const ContinuousIndexType & cindex,
        WeightsType & derivativeWeights,
        IndexType & startIndex) const
{
    unsigned int j = 0;
    for( unsigned int i = 0; i < SpaceDimension; ++i )
    {
      double x = cindex[ i ] - static_cast< double >( startIndex[ i ] );
      for( unsigned int k = 0; k < this->m_SupportSize[ i ]; ++k )
      {
          derivativeWeights[ j ] = this->m_DerivativeKernel->Evaluate( x );
          x-= 1.0;
          ++j;
      }
    }
}//end EvaluateDerivative
} // end namespace itk
#endif
