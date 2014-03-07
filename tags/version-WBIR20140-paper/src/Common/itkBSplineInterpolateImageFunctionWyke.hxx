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
/*=========================================================================
 *
 *  Portions of this file are subject to the VTK Toolkit Version 3 copyright.
 *
 *  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
 *
 *  For complete copyright, license and disclaimer of warranty information
 *  please refer to the NOTICE file at the top of the ITK source tree.
 *
 *=========================================================================*/
#ifndef __itkBSplineInterpolateImageFunctionWyke_hxx
#define __itkBSplineInterpolateImageFunctionWyke_hxx

#include "itkBSplineInterpolateImageFunctionWyke.h"
#include "itkImageLinearIteratorWithIndex.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageRegionIterator.h"

#include "itkVector.h"

#include "itkMatrix.h"

namespace itk
{
/**
 * Constructor
 */
template< class TImageType, class TCoordRep, class TCoefficientType, unsigned int splineOrder >
BSplineInterpolateImageFunctionWyke< TImageType, TCoordRep, TCoefficientType, splineOrder >
::BSplineInterpolateImageFunctionWyke()
{
    m_NumberOfThreads = 1;
    m_ThreadedEvaluateIndex = NULL;
    m_ThreadedWeights = NULL;
    m_ThreadedWeightsDerivative = NULL;
    this->m_UseImageDirection = true;

    m_CoefficientFilter = CoefficientFilter::New();
    m_CoefficientFilter->SetSplineOrder(splineOrder);

    m_Coefficients = CoefficientImageType::New();

    if(splineOrder>5)
    {
        // SplineOrder not implemented yet.
        ExceptionObject err(__FILE__, __LINE__);
        err.SetLocation(ITK_LOCATION);
        err.SetDescription("SplineOrder must be between 0 and 5. Requested spline order has not been implemented yet.");
        throw err;
    }

    m_MaxNumberInterpolationPoints = 1;
    for ( unsigned int n = 0; n < ImageDimension; n++ )
    {
        m_MaxNumberInterpolationPoints *= ( splineOrder + 1 );
    }
    this->SetThreads();

}

template< class TImageType, class TCoordRep, class TCoefficientType, unsigned int splineOrder >
BSplineInterpolateImageFunctionWyke< TImageType, TCoordRep, TCoefficientType, splineOrder >
::~BSplineInterpolateImageFunctionWyke()
{
    if ( m_ThreadedEvaluateIndex != NULL )
    {
        delete[] m_ThreadedEvaluateIndex;
        m_ThreadedEvaluateIndex = NULL;
    }
    if ( m_ThreadedWeights != NULL )
    {
        delete[] m_ThreadedWeights;
        m_ThreadedWeights = NULL;
    }
    if ( m_ThreadedWeightsDerivative != NULL )
    {
        delete[] m_ThreadedWeightsDerivative;
        m_ThreadedWeightsDerivative = NULL;
    }
}

/**
 * Standard "PrintSelf" method
 */
template< class TImageType, class TCoordRep, class TCoefficientType, unsigned int splineOrder >
void
BSplineInterpolateImageFunctionWyke< TImageType, TCoordRep, TCoefficientType, splineOrder >
::PrintSelf(std::ostream & os,Indent indent) const
{
    Superclass::PrintSelf(os, indent);
    os << indent << "Spline Order: " << splineOrder << std::endl;
    os << indent << "UseImageDirection = "
       << ( this->m_UseImageDirection ? "On" : "Off" ) << std::endl;
    os << indent << "NumberOfThreads: " << m_NumberOfThreads  << std::endl;
}

template< class TImageType, class TCoordRep, class TCoefficientType, unsigned int splineOrder >
void
BSplineInterpolateImageFunctionWyke< TImageType, TCoordRep, TCoefficientType, splineOrder >
::SetInputImage(const TImageType *inputData)
{
    if ( inputData )
    {
        Superclass::SetInputImage(inputData);

        m_CoefficientFilter->SetInput(inputData);
        m_CoefficientFilter->Update();
        m_Coefficients = m_CoefficientFilter->GetOutput();
        m_DataLength = inputData->GetBufferedRegion().GetSize();

        for(unsigned int n = 0; n < ImageDimension; ++n)
        {
            m_OffsetTable[n] = m_Coefficients->GetOffsetTable()[n];
        }

        m_Spacing = inputData->GetSpacing();
        m_StartIndex = this->GetStartIndex();
        m_EndIndex = this->GetEndIndex();
    }
    else
    {
        m_Coefficients = NULL;
    }
}

template< class TImageType, class TCoordRep, class TCoefficientType, unsigned int splineOrder >
void
BSplineInterpolateImageFunctionWyke< TImageType, TCoordRep, TCoefficientType, splineOrder >
::SetNumberOfThreads(ThreadIdType numThreads)
{
    m_NumberOfThreads = numThreads;
    this->SetThreads();
}

template< class TImageType, class TCoordRep, class TCoefficientType, unsigned int splineOrder >
typename
BSplineInterpolateImageFunctionWyke< TImageType, TCoordRep, TCoefficientType, splineOrder >
::OutputType
BSplineInterpolateImageFunctionWyke< TImageType, TCoordRep, TCoefficientType, splineOrder >
::EvaluateAtContinuousIndex(const ContinuousIndexType & x) const
{
    // Allocate memory on the stack
    long evaluateIndexData[(splineOrder+1)*ImageDimension];
    long stepsData[(splineOrder+1)*ImageDimension];
    double weightsData[(splineOrder+1)*ImageDimension];

    vnl_matrix_ref<long> evaluateIndex(ImageDimension,splineOrder+1,evaluateIndexData);
    double * weights = &(weightsData[0]);
    long * steps = &(stepsData[0]);

    this->DetermineRegionOfSupport(evaluateIndex, x);
    SetInterpolationWeights( x, evaluateIndex, weights);
    this->ApplyMirrorBoundaryConditions(evaluateIndex);

    OutputType interpolated = 0.0;

    //Calculate steps for image pointer
    for (unsigned int n = 0; n < ImageDimension; ++n )
    {
        for( unsigned int k = 0; k <= splineOrder; ++k)
        {
            steps[(splineOrder+1)*n+k] = (evaluateIndex[n][k])*m_OffsetTable[n];
        }
    }

    //Call recursive sampling function
    interpolated = sampleFunction< ImageDimension, splineOrder, TCoordRep >
            ::sample( (m_Coefficients->GetBufferPointer()),
                      steps,
                      weights);

    return interpolated;
}

template< class TImageType, class TCoordRep, class TCoefficientType, unsigned int splineOrder >
void
BSplineInterpolateImageFunctionWyke< TImageType, TCoordRep, TCoefficientType, splineOrder >
::EvaluateValueAndDerivativeAtContinuousIndex(const ContinuousIndexType & x,
                                              OutputType & value,
                                              CovariantVectorType & derivative
                                              ) const
{
    // Allocate memory on the stack
    long evaluateIndexData[(splineOrder+1)*ImageDimension];
    long stepsData[(splineOrder+1)*ImageDimension];
    double weightsData[(splineOrder+1)*ImageDimension];
    double derivativeWeightsData[(splineOrder+1)*ImageDimension];

    vnl_matrix_ref<long> evaluateIndex(ImageDimension,splineOrder+1,evaluateIndexData);
    double * weights = &(weightsData[0]);
    double * derivativeWeights = &(derivativeWeightsData[0]);
    long * steps = &(stepsData[0]);

    this->DetermineRegionOfSupport(evaluateIndex, x);
    SetInterpolationWeights( x, evaluateIndex, weights);
    SetDerivativeWeights(x, evaluateIndex, derivativeWeights);
    this->ApplyMirrorBoundaryConditions(evaluateIndex);

    const InputImageType *inputImage = this->GetInputImage();

    //Calculate steps for coefficients pointer
    for (unsigned int n = 0; n < ImageDimension; ++n )
    {
        for( unsigned int k = 0; k <= splineOrder; ++k)
        {
            steps[(splineOrder+1)*n+k] = (evaluateIndex[n][k])*m_OffsetTable[n];
        }
    }

    //Call recursive sampling function
    TCoordRep derivativeValue[ImageDimension+1];
    sampleFunction< ImageDimension, splineOrder, TCoordRep >
            ::sampleValueAndDerivative(derivativeValue,
                               m_Coefficients->GetBufferPointer(),
                               steps,
                               weights,
                               derivativeWeights);

    for(unsigned int n = 0; n < ImageDimension; ++n)
    {
        derivative[n] = derivativeValue[n+1] / m_Spacing[n];
    }

    value = derivativeValue[0];


    if ( this->m_UseImageDirection )
    {
        CovariantVectorType orientedDerivative;
        inputImage->TransformLocalVectorToPhysicalVector(derivative, orientedDerivative);
        derivative = orientedDerivative;
    }
}

template< class TImageType, class TCoordRep, class TCoefficientType, unsigned int splineOrder >
typename
BSplineInterpolateImageFunctionWyke< TImageType, TCoordRep, TCoefficientType, splineOrder >
::CovariantVectorType
BSplineInterpolateImageFunctionWyke< TImageType, TCoordRep, TCoefficientType, splineOrder >
::EvaluateDerivativeAtContinuousIndex(const ContinuousIndexType & x) const
{

    // Allocate memory on the stack
    long evaluateIndexData[(splineOrder+1)*ImageDimension];
    long stepsData[(splineOrder+1)*ImageDimension];
    double weightsData[(splineOrder+1)*ImageDimension];
    double derivativeWeightsData[(splineOrder+1)*ImageDimension];

    vnl_matrix_ref<long> evaluateIndex(ImageDimension,splineOrder+1,evaluateIndexData);
    double * weights = &(weightsData[0]);
    double * derivativeWeights = &(derivativeWeightsData[0]);
    long * steps = &(stepsData[0]);

    this->DetermineRegionOfSupport(evaluateIndex, x);
    SetInterpolationWeights( x, evaluateIndex, weights);
    SetDerivativeWeights(x, evaluateIndex, derivativeWeights);
    this->ApplyMirrorBoundaryConditions(evaluateIndex);

    const InputImageType *inputImage = this->GetInputImage();

    //Calculate steps for coefficients pointer
    for (unsigned int n = 0; n < ImageDimension; ++n )
    {
        for( unsigned int k = 0; k <= splineOrder; ++k)
        {
            steps[(splineOrder+1)*n+k] = (evaluateIndex[n][k])*m_OffsetTable[n];
        }
    }

    //Call recursive sampling function
    TCoordRep derivativeValue[ImageDimension+1];
    sampleFunction< ImageDimension, splineOrder, TCoordRep >
            ::sampleValueAndDerivative(derivativeValue,
                               m_Coefficients->GetBufferPointer(),
                               steps,
                               weights,
                               derivativeWeights);

    CovariantVectorType derivative;

    for(unsigned int n = 0; n < ImageDimension; ++n)
    {
        derivative[n] = derivativeValue[n+1] / m_Spacing[n];
    }


    if ( this->m_UseImageDirection )
    {
        CovariantVectorType orientedDerivative;
        inputImage->TransformLocalVectorToPhysicalVector(derivative, orientedDerivative);
        return orientedDerivative;
    }

    return derivative;
}


template< class TImageType, class TCoordRep, class TCoefficientType, unsigned int splineOrder >
void
BSplineInterpolateImageFunctionWyke< TImageType, TCoordRep, TCoefficientType, splineOrder >
::SetInterpolationWeights(const ContinuousIndexType & x,
                          const vnl_matrix< long > & evaluateIndex,
                          double * weights) const
{
    itk::Vector<double, splineOrder+1> weightsvec;
    const int idx = Math::Floor<int>( splineOrder / 2.0 );
    weightsvec.Fill( 0.0 );


    for ( unsigned int n = 0; n < ImageDimension; n++ )
    {
        double w = x[n] - (double)evaluateIndex[n][idx];
        BSplineWeights<splineOrder,TCoefficientType>::getWeights(weightsvec,w);
        for(unsigned int k = 0; k <= splineOrder; ++k)
        {
            weights[(splineOrder+1)*n+k] = weightsvec[k];
        }
        weightsvec.Fill( 0.0 );
    }
}

template< class TImageType, class TCoordRep, class TCoefficientType, unsigned int splineOrder >
void
BSplineInterpolateImageFunctionWyke< TImageType, TCoordRep, TCoefficientType, splineOrder >
::SetDerivativeWeights(const ContinuousIndexType & x,
                       const vnl_matrix< long > & evaluateIndex,
                       double * weights) const
{
    itk::Vector<double, splineOrder+1> weightsvec;

    const int idx = Math::Floor<int>( splineOrder / 2.0 )+1;

    for ( unsigned int n = 0; n < ImageDimension; n++ )
    {
        weightsvec.Fill( 0.0 );
        const double w = x[n] - (double)evaluateIndex[n][idx] + 0.5;
        BSplineWeights<splineOrder,TCoefficientType>::getDerivativeWeights(weightsvec, w);

        for(unsigned int k = 0; k <= splineOrder; ++k)
        {
            weights[(splineOrder+1)*n+k] = weightsvec[k];
        }
    }
}

//template< class TImageType, class TCoordRep, class TCoefficientType, unsigned int splineOrder >
//void
//BSplineInterpolateImageFunctionWyke< TImageType, TCoordRep, TCoefficientType, splineOrder >
//::SetHessianWeights(const ContinuousIndexType & x,
//                    const vnl_matrix< long > & evaluateIndex,
//                    double * weights) const
//{
//    itk::Vector<double, splineOrder+1> weightsvec;
//    weightsvec.Fill( 0.0 );

//    for ( unsigned int n = 0; n < ImageDimension; n++ )
//    {
//        int idx = floor( splineOrder / 2.0 );//FIX
//        double w = x[n] - (double)evaluateIndex[n][idx];
//        this->m_BSplineWeightInstance->getHessianWeights(weightsvec, w);
//        for(unsigned int k = 0; k <= splineOrder; ++k)
//        {
//            weights[(splineOrder+1)*n+k] = weightsvec[k];
//        }
//        weightsvec.Fill( 0.0 );
//    }
//}

template< class TImageType, class TCoordRep, class TCoefficientType, unsigned int splineOrder >
void
BSplineInterpolateImageFunctionWyke< TImageType, TCoordRep, TCoefficientType, splineOrder >
::SetThreads()
{
    if ( m_ThreadedEvaluateIndex != NULL )
    {
        delete[] m_ThreadedEvaluateIndex;
    }
    m_ThreadedEvaluateIndex = new vnl_matrix< long >[m_NumberOfThreads];
    if ( m_ThreadedWeights != NULL )
    {
        delete[] m_ThreadedWeights;
    }
    m_ThreadedWeights = new vnl_matrix< double >[m_NumberOfThreads];
    if ( m_ThreadedWeightsDerivative != NULL )
    {
        delete[] m_ThreadedWeightsDerivative;
    }
    m_ThreadedWeightsDerivative = new vnl_matrix< double >[m_NumberOfThreads];
    for ( unsigned int i = 0; i < m_NumberOfThreads; i++ )
    {
        m_ThreadedEvaluateIndex[i].set_size(ImageDimension, splineOrder + 1);
        m_ThreadedWeights[i].set_size(ImageDimension, splineOrder + 1);
        m_ThreadedWeightsDerivative[i].set_size(ImageDimension, splineOrder + 1);
    }
}

template< class TImageType, class TCoordRep, class TCoefficientType, unsigned int splineOrder >
void
BSplineInterpolateImageFunctionWyke< TImageType, TCoordRep, TCoefficientType, splineOrder >
::DetermineRegionOfSupport(vnl_matrix< long > & evaluateIndex, const ContinuousIndexType & x) const
{
    const float halfOffset = splineOrder & 1 ? 0.0 : 0.5;
    for ( unsigned int n = 0; n < ImageDimension; n++ )
    {
        long indx = Math::Floor<long>( (float)x[n] + halfOffset ) - splineOrder / 2;
        for ( unsigned int k = 0; k <= splineOrder; k++ )
        {
            evaluateIndex[n][k] = indx++;
        }
    }
}

template< class TImageType, class TCoordRep, class TCoefficientType, unsigned int splineOrder >
void
BSplineInterpolateImageFunctionWyke< TImageType, TCoordRep, TCoefficientType, splineOrder >
::ApplyMirrorBoundaryConditions(vnl_matrix< long > & evaluateIndex) const
{

    for ( unsigned int n = 0; n < ImageDimension; n++ )
    {
       if ( m_DataLength[n] == 1 )
        {
            for ( unsigned int k = 0; k <= splineOrder; k++ )
            {
                evaluateIndex[n][k] = 0;
            }
        }
        else
        {
            for ( unsigned int k = 0; k <= splineOrder; k++ )
            {
                if ( evaluateIndex[n][k] < m_StartIndex[n] )
                {
                    evaluateIndex[n][k] = m_StartIndex[n] +
                            ( m_StartIndex[n] - evaluateIndex[n][k] );
                }
                if ( evaluateIndex[n][k] >= m_EndIndex[n] )
                {
                    evaluateIndex[n][k] = m_EndIndex[n] -
                            ( evaluateIndex[n][k] - m_EndIndex[n] );
                }
            }
        }
    }
}
} // namespace itk

#endif
