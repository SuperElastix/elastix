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
#ifndef __itkPointToSurfaceDistanceMetric_h
#define __itkPointToSurfaceDistanceMetric_h

#include "itkSingleValuedPointSetToPointSetMetric.h"

/*These are included just for the metric. Presence of these stuffs also signifies ad-hoc definition or being less generic of the metric*/
#include "itkDanielssonDistanceMapImageFilter.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkInvertIntensityImageFilter.h"
#include "itkAddImageFilter.h"
#include "itkBSplineInterpolateImageFunction.h"
/*These are included just for the metric. Presence of these stuffs also signifies ad-hoc definition of the metric*/

namespace itk
{

template< class TFixedPointSet, class TMovingPointSet >
class PointToSurfaceDistanceMetric :
        public SingleValuedPointSetToPointSetMetric< TFixedPointSet, TMovingPointSet >
{
public:
    using Self = PointToSurfaceDistanceMetric;
    using Superclass = SingleValuedPointSetToPointSetMetric< TFixedPointSet, TMovingPointSet >;
    using Pointer = SmartPointer< Self >;
    using ConstPointer = SmartPointer< const Self >;

    /** Types transferred from the base class */
    using TransformPointer = typename Superclass::TransformPointer;
    using TransformParametersType = typename Superclass::TransformParametersType;
    using TransformJacobianType = typename Superclass::TransformJacobianType;
    using MeasureType = typename Superclass::MeasureType;
    using DerivativeType = typename Superclass::DerivativeType;
    using DerivativeValueType = typename Superclass::DerivativeValueType;
    using FixedPointSetType = typename Superclass::FixedPointSetType;
    using MovingPointSetType = typename Superclass::MovingPointSetType;
    using FixedPointSetConstPointer = typename Superclass::FixedPointSetConstPointer;
    using MovingPointSetConstPointer = typename Superclass::MovingPointSetConstPointer;
    using PointIterator = typename Superclass::PointIterator;
    using PointDataIterator = typename Superclass::PointDataIterator;
    using InputPointType = typename Superclass::InputPointType;
    using OutputPointType = typename Superclass::OutputPointType;
    using CoordRepType = typename OutputPointType::CoordRepType;
    using VnlVectorType = vnl_vector< CoordRepType >;
    using NonZeroJacobianIndicesType = typename Superclass::NonZeroJacobianIndicesType;

	   /** Constants for the pointset dimensions. */
    itkStaticConstMacro( FixedPointSetDimension, unsigned int, TFixedPointSet::PointDimension );

    using ImageType = itk::Image< float, itkGetStaticConstMacro( FixedPointSetDimension )>;
    using InputImageType = itk::Image< unsigned char,  itkGetStaticConstMacro( FixedPointSetDimension ) >;
    using RescalerType = itk::RescaleIntensityImageFilter< ImageType, ImageType >;
    using DtFilterType = itk::DanielssonDistanceMapImageFilter< InputImageType, ImageType, ImageType >;
    using NegateFilterType = itk::InvertIntensityImageFilter< InputImageType, InputImageType >;
    using AddImageFilt = itk::AddImageFilter< ImageType, ImageType, ImageType >;
    using SegReaderType =itk::ImageFileReader< InputImageType >;
    using DTReaderType = itk::ImageFileReader< ImageType  >;
    using WriterType = itk::ImageFileWriter< ImageType >;
    using InterpolatorType = itk::BSplineInterpolateImageFunction< ImageType, float, float >;
    using ConstDTimage = typename ImageType::ConstPointer;
	 /** Method for creation through the object factory. */
    itkNewMacro( Self );//OK

    /** Run-time type information (and related methods). */
    itkTypeMacro( PointToSurfaceDistanceMetric, SingleValuedPointSetToPointSetMetric );//OK
    /**  Initialize. */
    virtual void Initialize();

    /**  Get the value for single valued optimizers. */
    MeasureType GetValue( const TransformParametersType & parameters ) const;

    /** Get the derivatives of the match measure. */
    void GetDerivative( const TransformParametersType & parameters, DerivativeType & Derivative ) const;

    /**  Get value and derivatives for multiple valued optimizers. */
    void GetValueAndDerivative( const TransformParametersType & parameters, MeasureType & Value, DerivativeType & Derivative ) const;

    /** Set input Segmented Image File **/
    void SetSegImageIn( const std::string  str ) ;

    /** Set input Segmented Image File **/
    void SetDTImageIn( const std::string  str ) ;

    /** Set output Distance Transform File **/
    void SetDTImageOut(  const std::string  str );

    /** Get Distance Transform Image*/
    ConstDTimage GetDTImage() const;

protected:

    PointToSurfaceDistanceMetric();
    virtual ~PointToSurfaceDistanceMetric() = default;
    bool m_AvPointWeigh{true};

private:

    PointToSurfaceDistanceMetric( const Self & );
    std::string m_segmentationFileIn, m_distanceTransformFileIn, m_distanceTransformFileOut;
    typename ImageType::Pointer m_internalDistanceTransformImage;
    typename InterpolatorType::Pointer m_interpolator;

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkPointToSurfaceDistanceMetric.hxx"
#endif

#endif
