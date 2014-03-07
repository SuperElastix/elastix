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
#ifndef __itkAdvancedInterpolateImageFunction_h
#define __itkAdvancedInterpolateImageFunction_h

#include "itkInterpolateImageFunction.h"

namespace itk
{

template< class TInputImage, class TCoordRep = double >
class ITK_EXPORT AdvancedInterpolateImageFunction:
        public InterpolateImageFunction< TInputImage,TCoordRep >
{
public:
    /** Standard class typedefs. */
    typedef AdvancedInterpolateImageFunction                Self;
    typedef InterpolateImageFunction< TInputImage, TCoordRep > Superclass;
    typedef SmartPointer< Self >                              Pointer;
    typedef SmartPointer< const Self >                        ConstPointer;

    itkTypeMacro(AdvancedInterpolateImageFunction, InterpolateImageFunction);

    typedef typename Superclass::OutputType OutputType;
    typedef typename Superclass::InputImageType InputImageType;
    itkStaticConstMacro(ImageDimension, unsigned int, Superclass::ImageDimension);
    typedef typename Superclass::PointType PointType;
    typedef typename Superclass::IndexType IndexType;
    typedef typename Superclass::IndexValueType IndexValueType;
    typedef typename Superclass::ContinuousIndexType ContinuousIndexType;
    typedef typename NumericTraits< typename TInputImage::PixelType >::RealType RealType;
    typedef CovariantVector< OutputType, itkGetStaticConstMacro(ImageDimension) >    CovariantVectorType;

    virtual OutputType Evaluate(const PointType & point) const
    {
      ContinuousIndexType index;

      this->GetInputImage()->TransformPhysicalPointToContinuousIndex(point, index);
      return ( this->EvaluateAtContinuousIndex(index) );
    }

    virtual OutputType Evaluate(const PointType & point, ThreadIdType threadID) const = 0;
    virtual OutputType EvaluateAtIndex(const IndexType & index) const
    {
      return ( static_cast< RealType >( this->GetInputImage()->GetPixel(index) ) );
    }

    virtual OutputType EvaluateAtContinuousIndex(const ContinuousIndexType & index) const = 0;
    virtual OutputType EvaluateAtContinuousIndex(const ContinuousIndexType & index, ThreadIdType threadID) const = 0;

    virtual CovariantVectorType EvaluateDerivative(const PointType & point) const = 0;
    virtual CovariantVectorType EvaluateDerivative(const PointType & point, ThreadIdType threadID) const = 0;
    virtual CovariantVectorType EvaluateDerivativeAtContinuousIndex(const ContinuousIndexType & x) const = 0;
    virtual CovariantVectorType EvaluateDerivativeAtContinuousIndex(const ContinuousIndexType & x,ThreadIdType threadID) const = 0;

    virtual void EvaluateValueAndDerivative(const PointType & point, OutputType & value,CovariantVectorType & deriv) const = 0;
    virtual void EvaluateValueAndDerivative(const PointType & point, OutputType & value,CovariantVectorType & deriv,ThreadIdType threadID) const = 0;
    //virtual void EvaluateValueAndDerivativeAtContinuousIndex(const ContinuousIndexType & x, OutputType & value,CovariantVectorType & deriv) const = 0;
    virtual void EvaluateValueAndDerivativeAtContinuousIndex(const ContinuousIndexType & x, OutputType & value,CovariantVectorType & deriv, ThreadIdType threadID) const = 0;

    virtual void SetNumberOfThreads(ThreadIdType numThreads)=0;

    AdvancedInterpolateImageFunction(){}
    ~AdvancedInterpolateImageFunction(){}

    void PrintSelf(std::ostream & os, Indent indent) const
    { Superclass::PrintSelf(os, indent); }


private:
    AdvancedInterpolateImageFunction(const Self &);
    void operator=(const Self &);
};
} // namespace itk

#endif
