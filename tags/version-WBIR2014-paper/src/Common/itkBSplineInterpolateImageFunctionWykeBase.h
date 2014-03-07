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
#ifndef __itkBSplineInterpolateImageFunctionWykeBase_h
#define __itkBSplineInterpolateImageFunctionWykeBase_h

#include <vector>

#include "itkAdvancedInterpolateImageFunction.h"
#include "itkInterpolateImageFunction.h"
#include "vnl/vnl_matrix.h"

#include "itkBSplineDecompositionImageFilter.h"
#include "itkConceptChecking.h"
#include "itkCovariantVector.h"

namespace itk
{

template< class TImageType,
          class TCoordRep = double,
          class TCoefficientType = double>
class ITK_EXPORT BSplineInterpolateImageFunctionWykeBase:
        public AdvancedInterpolateImageFunction< TImageType, TCoordRep >
{
public:
    /** Standard class typedefs. */
    typedef BSplineInterpolateImageFunctionWykeBase                     Self;
    typedef AdvancedInterpolateImageFunction< TImageType, TCoordRep >   Superclass;
    typedef SmartPointer< Self >                                        Pointer;
    typedef SmartPointer< const Self >                                  ConstPointer;

    itkTypeMacro(BSplineInterpolateImageFunctionWykeBase, AdvancedInterpolateImageFunction)
    itkNewMacro(Self)

    typedef typename Superclass::OutputType OutputType;
    typedef typename Superclass::InputImageType InputImageType;
    itkStaticConstMacro(ImageDimension, unsigned int, Superclass::ImageDimension);
    typedef typename Superclass::IndexType IndexType;
    typedef typename Superclass::ContinuousIndexType ContinuousIndexType;
    typedef typename Superclass::PointType PointType;
    typedef ImageLinearIteratorWithIndex< TImageType > Iterator;
    typedef TCoefficientType CoefficientDataType;
    typedef Image< CoefficientDataType,  itkGetStaticConstMacro(ImageDimension) >    CoefficientImageType;
    typedef BSplineDecompositionImageFilter< TImageType,    CoefficientImageType > CoefficientFilter;
    typedef typename CoefficientFilter::Pointer    CoefficientFilterPointer;
    typedef CovariantVector< OutputType, itkGetStaticConstMacro(ImageDimension) >    CovariantVectorType;

    OutputType Evaluate(const PointType & point) const;
    OutputType Evaluate(const PointType & point, ThreadIdType threadID) const;
    OutputType EvaluateAtContinuousIndex(const ContinuousIndexType & index) const;
    OutputType EvaluateAtContinuousIndex(const ContinuousIndexType & index, ThreadIdType threadID) const;

    CovariantVectorType EvaluateDerivative(const PointType & point) const;
    CovariantVectorType EvaluateDerivative(const PointType & point, ThreadIdType threadID) const;
    CovariantVectorType EvaluateDerivativeAtContinuousIndex(const ContinuousIndexType & x) const;
    CovariantVectorType EvaluateDerivativeAtContinuousIndex(const ContinuousIndexType & x,ThreadIdType threadID) const;

    void EvaluateValueAndDerivative(const PointType & point, OutputType & value,CovariantVectorType & deriv) const;
    void EvaluateValueAndDerivative(const PointType & point, OutputType & value,CovariantVectorType & deriv,ThreadIdType threadID) const;
    void EvaluateValueAndDerivativeAtContinuousIndex(const ContinuousIndexType & x, OutputType & value,CovariantVectorType & deriv) const;
    void EvaluateValueAndDerivativeAtContinuousIndex(const ContinuousIndexType & x, OutputType & value,CovariantVectorType & deriv, ThreadIdType threadID) const;

    void SetSplineOrder(unsigned int splineOrder);
    void SetInputImage(const InputImageType *inputData);
    void SetNumberOfThreads(ThreadIdType numThreads);

    itkGetConstMacro(SplineOrder, int)

protected:
    BSplineInterpolateImageFunctionWykeBase();
    ~BSplineInterpolateImageFunctionWykeBase(){}

    typename AdvancedInterpolateImageFunction<TImageType, TCoordRep>::Pointer m_interpolatorInstance;
    unsigned int m_SplineOrder;

    void PrintSelf(std::ostream & os, Indent indent) const;

    OutputType EvaluateAtContinuousIndexInternal(const ContinuousIndexType & index, vnl_matrix< long > & evaluateIndex, vnl_matrix< double > & weights) const
    {
        itkExceptionMacro ("Exception: the method 'EvaluateAtContinuousIndexInternal' is removed from this class.");

    }

    void EvaluateValueAndDerivativeAtContinuousIndexInternal(const ContinuousIndexType & x,OutputType & value, CovariantVectorType & derivativeValue, vnl_matrix< long > & evaluateIndex,vnl_matrix< double > & weights,  vnl_matrix< double > & weightsDerivative) const
    {
        itkExceptionMacro ("Exception: the method 'EvaluateValueAndDerivativeAtContinuousIndexInternal' is removed from this class.");
    }

    CovariantVectorType EvaluateDerivativeAtContinuousIndexInternal(const ContinuousIndexType & x, vnl_matrix< long > & evaluateIndex, vnl_matrix< double > & weights, vnl_matrix< double > & weightsDerivative) const
    {
        itkExceptionMacro ("Exception: the method 'EvaluateDerivativeAtContinuousIndexInternal' is removed from this class.");
    }


private:
    BSplineInterpolateImageFunctionWykeBase(const Self &);
    void operator=(const Self &);
};

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkBSplineInterpolateImageFunctionWykeBase.hxx"
#endif

#endif
