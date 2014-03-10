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
#ifndef __itkAdvancedInterpolateImageFunction_h
#define __itkAdvancedInterpolateImageFunction_h

#include "itkInterpolateImageFunction.h"

namespace itk
{

template< class TInputImage, class TCoordRep = double >
class ITK_EXPORT AdvancedInterpolateImageFunction:
        public InterpolateImageFunction< TInputImage,TCoordRep >
{
    /** \class AdvancedInterpolateImageFunction
     * \brief Interpolate image function base class, with added methods.
     *
     * AdvancedInterpolateImageFunction is the advanced base for the
     * RecursiveBSplineInterpolateImageFunction class. This class is templated over
     * the input image type and the coordinate representation type
     * (e.g. float or double ). This class inherits from the
     * InterpolateImageFunction class.
     *
     * AdvancedInterpolateImageFunction
     * \ingroup ImageFunctions ImageInterpolators
     * \ingroup ITKImageFunction
     *
     */

public:
    /** Standard class typedefs. */
    typedef AdvancedInterpolateImageFunction                Self;
    typedef InterpolateImageFunction< TInputImage, TCoordRep > Superclass;
    typedef SmartPointer< Self >                              Pointer;
    typedef SmartPointer< const Self >                        ConstPointer;

    /** Run-time type information (and related methods). */
    itkTypeMacro(AdvancedInterpolateImageFunction, InterpolateImageFunction);

    /** OutputType typedef support. */
    typedef typename Superclass::OutputType OutputType;
    typedef CovariantVector< OutputType, itkGetStaticConstMacro(ImageDimension) >    CovariantVectorType;

    /** InputImageType typedef support. */
    typedef typename Superclass::InputImageType InputImageType;

    /** Dimension underlying input image. */
    itkStaticConstMacro(ImageDimension, unsigned int, Superclass::ImageDimension);

    /** Point typedef support. */
    typedef typename Superclass::PointType PointType;

    /** Index typedef support. */
    typedef typename Superclass::IndexType IndexType;
    typedef typename Superclass::IndexValueType IndexValueType;

    /** ContinuousIndex typedef support. */
    typedef typename Superclass::ContinuousIndexType ContinuousIndexType;

    /** RealType typedef support. */
    typedef typename NumericTraits< typename TInputImage::PixelType >::RealType RealType;

    /** Interpolate the image at a point position
     *
     * Returns the interpolated image intensity at a
     * specified point position. No bounds checking is done.
     * The point is assume to lie within the image buffer.
     *
     * ImageFunction::IsInsideBuffer() can be used to check bounds before
     * calling the method. */
    virtual OutputType Evaluate(const PointType & point) const
    {
      ContinuousIndexType index;

      this->GetInputImage()->TransformPhysicalPointToContinuousIndex(point, index);
      return ( this->EvaluateAtContinuousIndex(index) );
    }

    /** Interpolate the image at a continuous index position
     *
     * Returns the interpolated image intensity at a
     * specified index position. No bounds checking is done.
     * The point is assume to lie within the image buffer.
     *
     * Subclasses must override this method.
     *
     * ImageFunction::IsInsideBuffer() can be used to check bounds before
     * calling the method. */
    virtual OutputType Evaluate(const PointType & point, ThreadIdType threadID) const = 0;

    /** Interpolate the image at an index position.
     *
     * Simply returns the image value at the
     * specified index position. No bounds checking is done.
     * The point is assume to lie within the image buffer.
     *
     * ImageFunction::IsInsideBuffer() can be used to check bounds before
     * calling the method. */
    virtual OutputType EvaluateAtIndex(const IndexType & index) const
    {
      return ( static_cast< RealType >( this->GetInputImage()->GetPixel(index) ) );
    }

    /** Interpolate the image at an index position.
     *
     * The virtual methods below are necessary for the recursiveBSplineInterpolateImageFunction class.
     *
     */
    virtual OutputType EvaluateAtContinuousIndex(const ContinuousIndexType & index) const = 0;
    virtual OutputType EvaluateAtContinuousIndex(const ContinuousIndexType & index, ThreadIdType threadID) const = 0;

    virtual CovariantVectorType EvaluateDerivative(const PointType & point) const = 0;
    virtual CovariantVectorType EvaluateDerivative(const PointType & point, ThreadIdType threadID) const = 0;
    virtual CovariantVectorType EvaluateDerivativeAtContinuousIndex(const ContinuousIndexType & x) const = 0;
    virtual CovariantVectorType EvaluateDerivativeAtContinuousIndex(const ContinuousIndexType & x,ThreadIdType threadID) const = 0;

    virtual void EvaluateValueAndDerivative(const PointType & point, OutputType & value,CovariantVectorType & deriv) const = 0;
    virtual void EvaluateValueAndDerivative(const PointType & point, OutputType & value,CovariantVectorType & deriv,ThreadIdType threadID) const = 0;
    virtual void EvaluateValueAndDerivativeAtContinuousIndex(const ContinuousIndexType & x, OutputType & value,CovariantVectorType & deriv) const = 0;
    virtual void EvaluateValueAndDerivativeAtContinuousIndex(const ContinuousIndexType & x, OutputType & value,CovariantVectorType & deriv, ThreadIdType threadID) const = 0;

    virtual void SetNumberOfThreads(ThreadIdType numThreads)=0;

    AdvancedInterpolateImageFunction(){}
    ~AdvancedInterpolateImageFunction(){}

    void PrintSelf(std::ostream & os, Indent indent) const
    { Superclass::PrintSelf(os, indent); }


private:
    AdvancedInterpolateImageFunction(const Self &); //purposely not implemented
    void operator=(const Self &);                   //purposely not implemented
};
} // namespace itk

#endif
