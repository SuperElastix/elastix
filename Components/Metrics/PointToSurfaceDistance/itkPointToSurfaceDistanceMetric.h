/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkPointToSurfaceDistanceMetric_h
#define __itkPointToSurfaceDistanceMetric_h

#include "itkSingleValuedPointSetToPointSetMetric.h"
#include <cmath>

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
        typedef PointToSurfaceDistanceMetric Self;

	typedef SingleValuedPointSetToPointSetMetric<
    TFixedPointSet, TMovingPointSet >               Superclass;
    typedef SmartPointer< Self >       Pointer;
    typedef SmartPointer< const Self > ConstPointer;

    /** Types transferred from the base class */
    typedef typename Superclass::TransformType				TransformType;
    typedef typename Superclass::TransformPointer			TransformPointer;
    typedef typename Superclass::TransformParametersType                TransformParametersType;
    typedef typename Superclass::TransformJacobianType                  TransformJacobianType;
    typedef typename Superclass::MeasureType                            MeasureType;
    typedef typename Superclass::DerivativeType                         DerivativeType;
    typedef typename Superclass::DerivativeValueType                    DerivativeValueType;
    typedef typename Superclass::FixedPointSetType                      FixedPointSetType;
    typedef typename Superclass::MovingPointSetType                     MovingPointSetType;
    typedef typename Superclass::FixedPointSetConstPointer              FixedPointSetConstPointer;
    typedef typename Superclass::MovingPointSetConstPointer             MovingPointSetConstPointer;
    typedef typename Superclass::PointIterator				PointIterator;
    typedef typename Superclass::PointDataIterator			PointDataIterator;
    typedef typename Superclass::InputPointType				InputPointType;
    typedef typename Superclass::OutputPointType			OutputPointType;
    typedef typename OutputPointType::CoordRepType			CoordRepType;
    typedef vnl_vector< CoordRepType >					VnlVectorType;
    typedef typename Superclass::NonZeroJacobianIndicesType             NonZeroJacobianIndicesType;//OK

	   /** Constants for the pointset dimensions. */
    itkStaticConstMacro( FixedPointSetDimension, unsigned int, TFixedPointSet::PointDimension );

    typedef itk::Image< float, itkGetStaticConstMacro( FixedPointSetDimension )  >          ImageType;
    typedef itk::Image< float, itkGetStaticConstMacro( FixedPointSetDimension )  >          ImageType2;
    typedef itk::Image< unsigned char,  itkGetStaticConstMacro( FixedPointSetDimension )  > InputImageType;
    typedef itk::DanielssonDistanceMapImageFilter< InputImageType, ImageType, ImageType >   FilterType;
    typedef itk::RescaleIntensityImageFilter< ImageType, ImageType >                        RescalerType;
    typedef itk::DanielssonDistanceMapImageFilter< InputImageType, ImageType, ImageType >   DtFilterType;
    typedef itk::InvertIntensityImageFilter< InputImageType, InputImageType >               NegateFilterType;
    typedef itk::AddImageFilter< ImageType, ImageType, ImageType >                          AddImageFilt;
    typedef itk::ImageFileReader< InputImageType  >                                         SegReaderType;
    typedef itk::ImageFileReader< ImageType  >                                              DTReaderType;
    typedef itk::ImageFileWriter< ImageType >                                               WriterType;
    typedef itk::BSplineInterpolateImageFunction< ImageType, float, float>                  InterpolatorType;

    typename ImageType::IndexType Index;
    typename ImageType::PointType Point;
    typename DtFilterType::Pointer m_DTfilter,m_DTfilter2;
    typename NegateFilterType::Pointer m_NegateFilter;
    typename RescalerType::Pointer m_scaler;
    typename AddImageFilt::Pointer m_AddImage;
    typename SegReaderType::Pointer m_segReader,m_segReader2;
    typename DTReaderType::Pointer m_dtReader;
    typename WriterType::Pointer m_writer;
    typename ImageType::Pointer ADTimage;
    typename InterpolatorType::Pointer interpolator;
    bool     m_AvPointWeigh;//Added 05/2/16

	 /** Method for creation through the object factory. */
    itkNewMacro( Self );//OK

    /** Run-time type information (and related methods). */
    itkTypeMacro( PointToSurfaceDistanceMetric, SingleValuedPointSetToPointSetMetric );//OK
    /**  Initialize. */
    virtual void Initialize( void ) throw ( ExceptionObject );

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
    typename ImageType::ConstPointer GetDTImage(void);

protected:

    PointToSurfaceDistanceMetric();
    virtual ~PointToSurfaceDistanceMetric() {}

private:
    PointToSurfaceDistanceMetric( const Self & );
    std::string m_SegFileIn,m_DTFileIn,m_DTFileOut;
    //void operator=( const Self & );
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkPointToSurfaceDistanceMetric.hxx"
#endif

#endif
