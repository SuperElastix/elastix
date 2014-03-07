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
#ifndef __itkBSplineInterpolateImageFunctionWykeBase_hxx
#define __itkBSplineInterpolateImageFunctionWykeBase_hxx

#include "itkBSplineInterpolateImageFunctionWykeBase.h"
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
template< class TImageType, class TCoordRep, class TCoefficientType>
BSplineInterpolateImageFunctionWykeBase< TImageType, TCoordRep, TCoefficientType >
::BSplineInterpolateImageFunctionWykeBase()
{
    //Set the spline order
    unsigned int SplineOrder = 3;
    this->SetSplineOrder(SplineOrder);
}

template< class TImageType, class TCoordRep, class TCoefficientType >
void
BSplineInterpolateImageFunctionWykeBase< TImageType, TCoordRep, TCoefficientType >
::PrintSelf(std::ostream & os, Indent indent) const
{
    this->m_interpolatorInstance->PrintSelf(os, indent);
}

template< class TImageType, class TCoordRep, class TCoefficientType >
void
BSplineInterpolateImageFunctionWykeBase< TImageType, TCoordRep, TCoefficientType >
::SetSplineOrder(unsigned int splineOrder)
{
    if ( splineOrder == m_SplineOrder )
    {
        return;
    }
    m_SplineOrder = splineOrder;

    switch(m_SplineOrder)
    {
    case 0:
        m_interpolatorInstance = BSplineInterpolateImageFunctionWyke< TImageType, TCoordRep, TCoefficientType, 0 >::New();
        break;
    case 1:
        m_interpolatorInstance = BSplineInterpolateImageFunctionWyke< TImageType, TCoordRep, TCoefficientType, 1 >::New();
        break;
    case 2:
        m_interpolatorInstance = BSplineInterpolateImageFunctionWyke< TImageType, TCoordRep, TCoefficientType, 2 >::New();
        break;
    case 3:
        m_interpolatorInstance = BSplineInterpolateImageFunctionWyke< TImageType, TCoordRep, TCoefficientType, 3 >::New();
        break;
    case 4:
        m_interpolatorInstance = BSplineInterpolateImageFunctionWyke< TImageType, TCoordRep, TCoefficientType, 4 >::New();
        break;
    case 5:
        m_interpolatorInstance = BSplineInterpolateImageFunctionWyke< TImageType, TCoordRep, TCoefficientType, 5 >::New();

    }

}
template< class TImageType, class TCoordRep, class TCoefficientType >
void
BSplineInterpolateImageFunctionWykeBase< TImageType, TCoordRep, TCoefficientType >
::SetInputImage( const InputImageType * inputData )
{
    Superclass::SetInputImage(inputData);
    this->m_interpolatorInstance->SetInputImage(inputData);
}

template< class TImageType, class TCoordRep, class TCoefficientType >
typename BSplineInterpolateImageFunctionWykeBase< TImageType, TCoordRep, TCoefficientType >::OutputType
BSplineInterpolateImageFunctionWykeBase< TImageType, TCoordRep, TCoefficientType >::Evaluate( const PointType & point ) const
{
    return this->m_interpolatorInstance->Evaluate(point);
}

template< class TImageType, class TCoordRep, class TCoefficientType >
typename BSplineInterpolateImageFunctionWykeBase< TImageType, TCoordRep, TCoefficientType >::OutputType
BSplineInterpolateImageFunctionWykeBase< TImageType, TCoordRep, TCoefficientType >::Evaluate( const PointType & point, ThreadIdType threadID) const
{
    return this->m_interpolatorInstance->Evaluate(point, threadID);
}

template< class TImageType, class TCoordRep, class TCoefficientType >
typename BSplineInterpolateImageFunctionWykeBase< TImageType, TCoordRep, TCoefficientType >::OutputType
BSplineInterpolateImageFunctionWykeBase< TImageType, TCoordRep, TCoefficientType >::EvaluateAtContinuousIndex( const ContinuousIndexType & index) const
{
    return this->m_interpolatorInstance->EvaluateAtContinuousIndex(index);
}

template< class TImageType, class TCoordRep, class TCoefficientType >
typename BSplineInterpolateImageFunctionWykeBase< TImageType, TCoordRep, TCoefficientType >::OutputType
BSplineInterpolateImageFunctionWykeBase< TImageType, TCoordRep, TCoefficientType >::EvaluateAtContinuousIndex( const ContinuousIndexType & index, ThreadIdType threadID) const
{
    return this->m_interpolatorInstance->EvaluateAtContinuousIndex(index, threadID);
}

template< class TImageType, class TCoordRep, class TCoefficientType >
typename BSplineInterpolateImageFunctionWykeBase< TImageType, TCoordRep, TCoefficientType >::CovariantVectorType
BSplineInterpolateImageFunctionWykeBase< TImageType, TCoordRep, TCoefficientType >::EvaluateDerivative( const PointType & point ) const
{
    return this->m_interpolatorInstance->EvaluateDerivative(point );
}

template< class TImageType, class TCoordRep, class TCoefficientType >
typename BSplineInterpolateImageFunctionWykeBase< TImageType, TCoordRep, TCoefficientType >::CovariantVectorType
BSplineInterpolateImageFunctionWykeBase< TImageType, TCoordRep, TCoefficientType >::EvaluateDerivative( const PointType & point, ThreadIdType threadID ) const
{
    return this->m_interpolatorInstance->EvaluateDerivative(point, threadID );
}

template< class TImageType, class TCoordRep, class TCoefficientType >
typename BSplineInterpolateImageFunctionWykeBase< TImageType, TCoordRep, TCoefficientType >::CovariantVectorType
BSplineInterpolateImageFunctionWykeBase< TImageType, TCoordRep, TCoefficientType >::EvaluateDerivativeAtContinuousIndex(const ContinuousIndexType & x) const
{
    return this->m_interpolatorInstance->EvaluateDerivativeAtContinuousIndex( x );
}

template< class TImageType, class TCoordRep, class TCoefficientType >
typename BSplineInterpolateImageFunctionWykeBase< TImageType, TCoordRep, TCoefficientType >::CovariantVectorType
BSplineInterpolateImageFunctionWykeBase< TImageType, TCoordRep, TCoefficientType >::EvaluateDerivativeAtContinuousIndex( const ContinuousIndexType & point, ThreadIdType threadID ) const
{
    return this->m_interpolatorInstance->EvaluateDerivativeAtContinuousIndex(point, threadID );
}

template< class TImageType, class TCoordRep, class TCoefficientType >
void
BSplineInterpolateImageFunctionWykeBase< TImageType, TCoordRep, TCoefficientType >
::EvaluateValueAndDerivative(const PointType & point, OutputType & value,CovariantVectorType & deriv) const
{
    this->m_interpolatorInstance->EvaluateValueAndDerivative(point, value, deriv);
}

template< class TImageType, class TCoordRep, class TCoefficientType >
void
BSplineInterpolateImageFunctionWykeBase< TImageType, TCoordRep, TCoefficientType >
::EvaluateValueAndDerivativeAtContinuousIndex(const ContinuousIndexType & x, OutputType & value,CovariantVectorType & deriv) const
{
    this->m_interpolatorInstance->EvaluateValueAndDerivative(x, value, deriv);
}

template< class TImageType, class TCoordRep, class TCoefficientType >
void
BSplineInterpolateImageFunctionWykeBase< TImageType, TCoordRep, TCoefficientType >
::EvaluateValueAndDerivative(const PointType & point, OutputType & value,CovariantVectorType & deriv, ThreadIdType threadID) const
{
    this->m_interpolatorInstance->EvaluateValueAndDerivative(point, value, deriv, threadID);
}

template< class TImageType, class TCoordRep, class TCoefficientType >
void
BSplineInterpolateImageFunctionWykeBase< TImageType, TCoordRep, TCoefficientType >
::EvaluateValueAndDerivativeAtContinuousIndex(const ContinuousIndexType & x, OutputType & value,CovariantVectorType & deriv, ThreadIdType threadID) const
{
    this->m_interpolatorInstance->EvaluateValueAndDerivative(x, value, deriv, threadID);
}

template< class TImageType, class TCoordRep, class TCoefficientType >
void
BSplineInterpolateImageFunctionWykeBase< TImageType, TCoordRep, TCoefficientType >
::SetNumberOfThreads(ThreadIdType numThreads)
{
    this->m_interpolatorInstance->SetNumberOfThreads(numThreads);
}

} // namespace itk

#endif //itkBSplineInterpolateImageFunctionWykeBase_hxx
