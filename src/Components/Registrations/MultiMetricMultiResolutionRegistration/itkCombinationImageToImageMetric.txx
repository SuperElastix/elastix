#ifndef _itkCombinationImageToImageMetric_txx
#define _itkCombinationImageToImageMetric_txx

#include "itkCombinationImageToImageMetric.h"


/** Macros to reduce some copy-paste work.
 * These macros provide the implementation of
 * all Set/GetFixedImage, Set/GetInterpolator etc methods
 * 
 * The macros are undef'ed at the end of this file
 */

/** For setting objects, implement two methods */
#define itkImplementationSetObjectMacro(_name, _type) \
  template <class TFixedImage, class TMovingImage> \
    void \
    CombinationImageToImageMetric<TFixedImage,TMovingImage> \
    ::Set##_name ( _type *_arg, unsigned int pos ) \
  { \
    if (pos == 0) \
    { \
      this->Superclass::Set##_name ( _arg ); \
    } \
    ImageMetricType * testPtr = dynamic_cast<ImageMetricType *>( this->GetMetric(pos) ); \
    if ( testPtr ) \
    { \
      testPtr->Set##_name ( _arg ); \
    } \
  } \
  template <class TFixedImage, class TMovingImage> \
    void \
    CombinationImageToImageMetric<TFixedImage,TMovingImage> \
    ::Set##_name ( _type *_arg ) \
  { \
    for ( unsigned int i = 0; i < this->GetNumberOfMetrics(); i++ ) \
    { \
      this->Set##_name ( _arg, i); \
    } \
  }  // comments for allowing ; after calling the macro

/** For setting const objects, implement 2 methods */
#define itkImplementationSetConstObjectMacro(_name, _type) \
  template <class TFixedImage, class TMovingImage> \
    void \
    CombinationImageToImageMetric<TFixedImage,TMovingImage> \
    ::Set##_name ( const _type *_arg, unsigned int pos ) \
  { \
    if (pos == 0) \
    { \
    this->Superclass::Set##_name ( _arg ); \
    } \
    ImageMetricType * testPtr = dynamic_cast<ImageMetricType *>( this->GetMetric(pos) ); \
    if ( testPtr ) \
    { \
      testPtr->Set##_name ( _arg ); \
    } \
  } \
  template <class TFixedImage, class TMovingImage> \
    void \
    CombinationImageToImageMetric<TFixedImage,TMovingImage> \
    ::Set##_name ( const _type *_arg ) \
  { \
    for ( unsigned int i = 0; i < this->GetNumberOfMetrics(); i++ ) \
    { \
      this->Set##_name ( _arg, i); \
    } \
  }  // comment to allow ; after calling the macro

/** for getting const object, implement one method */
#define itkImplementationGetConstObjectMacro(_name,_type) \
  template <class TFixedImage, class TMovingImage> \
  const typename CombinationImageToImageMetric<TFixedImage,TMovingImage>:: \
    _type * CombinationImageToImageMetric<TFixedImage,TMovingImage> \
    ::Get##_name ( unsigned int pos ) const \
  { \
    const ImageMetricType * testPtr = dynamic_cast<const ImageMetricType * >( this->GetMetric(pos) ); \
    if ( testPtr ) \
    { \
      return testPtr->Get##_name (); \
    } \
    else \
    { \
    return 0 ; } }  //
 

namespace itk
{

  itkImplementationSetObjectMacro( Transform, TransformType );
  itkImplementationSetObjectMacro( Interpolator, InterpolatorType );
  itkImplementationSetObjectMacro( FixedImageMask, FixedImageMaskType );
  itkImplementationSetObjectMacro( MovingImageMask, MovingImageMaskType );

  itkImplementationSetConstObjectMacro( FixedImage, FixedImageType );
  itkImplementationSetConstObjectMacro( MovingImage, MovingImageType );

  itkImplementationGetConstObjectMacro( Transform, TransformType );
  itkImplementationGetConstObjectMacro( Interpolator, InterpolatorType );
  itkImplementationGetConstObjectMacro( FixedImageMask, FixedImageMaskType );
  itkImplementationGetConstObjectMacro( MovingImageMask, MovingImageMaskType );
  itkImplementationGetConstObjectMacro( FixedImage, FixedImageType );
  itkImplementationGetConstObjectMacro( MovingImage, MovingImageType );


  /**
	 * ********************* Constructor ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    CombinationImageToImageMetric<TFixedImage,TMovingImage>
    ::CombinationImageToImageMetric()
  {
    this->m_NumberOfMetrics = 0;
    this->ComputeGradientOff();
  
  } // end Constructor


  /**
	 * ********************* PrintSelf ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    CombinationImageToImageMetric<TFixedImage,TMovingImage>
    ::PrintSelf( std::ostream& os, Indent indent ) const
  {
		/** Call the superclass' PrintSelf. */
		Superclass::PrintSelf( os, indent );
		
		/** Add debugging information. */
		os << indent << "NumberOfMetrics: "
			<< this->m_NumberOfMetrics << std::endl;
    os << indent << "Metric pointer, weight, value:" << std::endl;
    for ( unsigned int i = 0; i < this->m_NumberOfMetrics; i++ )
    {
      os << indent << "Metric " << i << ": "
        << this->m_Metrics[ i ].GetPointer() << ", "
        << this->m_MetricWeights[ i ] << ", "
        << this->m_MetricValues[ i ] << std::endl;
    }
				
	} // end PrintSelf


  /**
	 * ******************** SetFixedImageRegion ************************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    CombinationImageToImageMetric<TFixedImage,TMovingImage>
    ::SetFixedImageRegion( const FixedImageRegionType _arg, unsigned int pos )
  {
    if ( pos == 0 )
    {
      this->Superclass::SetFixedImageRegion( _arg );
    }
    ImageMetricType * testPtr = dynamic_cast<ImageMetricType *>( this->GetMetric(pos) );
    if ( testPtr )
    {
      testPtr->SetFixedImageRegion( _arg );
    }
  } // end SetFixedImageRegion 


  /**
	 * ******************** SetFixedImageRegion ************************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    CombinationImageToImageMetric<TFixedImage,TMovingImage>
    ::SetFixedImageRegion( const FixedImageRegionType _arg )
  {
    for ( unsigned int i = 0; i < this->GetNumberOfMetrics(); i++ )
    { 
      this->SetFixedImageRegion( _arg, i);
    } 
  } // end SetFixedImageRegion

 
  /**
	 * ******************** GetFixedImageRegion ************************
	 */

  template <class TFixedImage, class TMovingImage>
    const typename CombinationImageToImageMetric<TFixedImage,TMovingImage>::
    FixedImageRegionType &
    CombinationImageToImageMetric<TFixedImage,TMovingImage>
    ::GetFixedImageRegion( unsigned int pos ) const
  {
    const ImageMetricType * testPtr = dynamic_cast<const ImageMetricType *>( this->GetMetric(pos) ); 
    if ( testPtr ) 
    { 
      return testPtr->GetFixedImageRegion(); 
    } 
    else 
    { 
      return this->m_NullFixedImageRegion; 
    }     
  } // end GetFixedImageRegion

 
  /**
	 * ********************* SetNumberOfMetrics ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    CombinationImageToImageMetric<TFixedImage,TMovingImage>
    ::SetNumberOfMetrics( unsigned int count )
  {
    if ( count != this->m_Metrics.size() )
    {
      this->m_NumberOfMetrics = count;
      this->m_Metrics.resize( count );
      this->m_MetricWeights.resize( count );
      this->m_MetricValues.resize( count );
      this->Modified();
    }

  } // end SetNumberOfMetrics


  /**
	 * ********************* SetMetric ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    CombinationImageToImageMetric<TFixedImage,TMovingImage>
    ::SetMetric( SingleValuedCostFunctionType * metric, unsigned int pos )
  {
    if ( pos >= this->GetNumberOfMetrics() )
    {
      this->SetNumberOfMetrics( pos+1 );
    }

    if ( metric != this->m_Metrics[ pos ] )
    {
      this->m_Metrics[ pos ] = metric;
      this->Modified();
    }
    
  } // end SetMetric
  

  /**
	 * ********************* GetMetric ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    typename CombinationImageToImageMetric<TFixedImage,TMovingImage>
    ::SingleValuedCostFunctionType * 
    CombinationImageToImageMetric<TFixedImage,TMovingImage>
    ::GetMetric( unsigned int pos ) const
  {
    if ( pos >= this->GetNumberOfMetrics() )
    {
      return 0;
    }
    else
    {
      return this->m_Metrics[ pos ];
    }
    
  } // end GetMetric


  /**
	 * ********************* SetMetricWeight ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    CombinationImageToImageMetric<TFixedImage,TMovingImage>
    ::SetMetricWeight( double weight, unsigned int pos )
  {
    if ( pos >= this->GetNumberOfMetrics() )
    {
      this->SetNumberOfMetrics( pos+1 );
    }

    if ( weight != this->m_MetricWeights[ pos ] )
    {
      this->m_MetricWeights[ pos ] = weight;
      this->Modified();
    }
    
  } // end SetMetricsWeight


  /**
	 * ********************* GetMetricWeight ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    double
    CombinationImageToImageMetric<TFixedImage,TMovingImage>
    ::GetMetricWeight( unsigned int pos ) const
  {
    if ( pos >= this->GetNumberOfMetrics() )
    {
      return 0.0;
    }
    else
    {
      return this->m_MetricWeights[ pos ];
    }
    
  } // end GetMetricWeight


  /**
	 * ********************* GetMetricValue ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    typename CombinationImageToImageMetric<TFixedImage,TMovingImage>::MeasureType
    CombinationImageToImageMetric<TFixedImage,TMovingImage>
    ::GetMetricValue( unsigned int pos ) const
  {
    if ( pos >= this->GetNumberOfMetrics() )
    {
      return 0.0;
    }
    else
    {
      return this->m_MetricValues[ pos ];
    }
    
  } // end GetMetricValue

  
  /**
	 * **************** GetNumberOfPixelsCounted ************************
	 */

  template <class TFixedImage, class TMovingImage>
    const unsigned long &
    CombinationImageToImageMetric<TFixedImage,TMovingImage>
    ::GetNumberOfPixelsCounted( void ) const
  {
    unsigned long sum = 0;
    for (unsigned int i=0; i < this->GetNumberOfMetrics(); ++i)
    {
      const ImageMetricType * testPtr =
        dynamic_cast<const ImageMetricType *>( this->GetMetric(i) );
      if ( testPtr )
      {
        sum += testPtr->GetNumberOfPixelsCounted();
      }
    }
    this->m_NumberOfPixelsCounted = sum;
    return this->m_NumberOfPixelsCounted;

  } // end GetNumberOfPixelsCounted


  /**
	 * ********************* Initialize ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    CombinationImageToImageMetric<TFixedImage,TMovingImage>
    ::Initialize( void ) throw ( ExceptionObject )
  {
    /** Check if transform, interpolator have been set. Effectively
     * this method checks if the first sub metric is set up completely.
     * This implicitly means that the first sub metric is an 
     * ImageToImageMetric, which is a reasonable demand.  */
    this->Superclass::Initialize();

    /** Check if at least one (image)metric is provided */
    if ( this->GetNumberOfMetrics() < 1 )
    {
      itkExceptionMacro( << "At least one metric should be set!" );
    }
    ImageMetricType * firstMetric = dynamic_cast<ImageMetricType *>(
      this->GetMetric(0) );
    if ( !firstMetric )
    {
      itkExceptionMacro( << "The first sub metric must be of type ImageToImageMetric!");
    }

    for ( unsigned int i = 0; i < this->GetNumberOfMetrics() ; i++ )
    {
      SingleValuedCostFunctionType * costfunc = this->GetMetric(i);
      if ( !costfunc )
      {
        itkExceptionMacro( << "Metric " << i << "has not been set!" );
      }
      ImageMetricType * testPtr = dynamic_cast<ImageMetricType *>( costfunc );
      if ( testPtr )
      {
        testPtr->Initialize();
      }
    }

    
  } // end Initialize

 
  /**
	 * ********************* GetValue ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    typename CombinationImageToImageMetric<TFixedImage,TMovingImage>::MeasureType
    CombinationImageToImageMetric<TFixedImage,TMovingImage>
    ::GetValue( const ParametersType & parameters ) const
  {
    /** Add all metric values. */
    MeasureType measure = NumericTraits< MeasureType >::Zero;
    for ( unsigned int i = 0; i < this->m_NumberOfMetrics; i++ )
    {
      this->m_MetricValues[ i ] =
        this->m_MetricWeights[ i ] * this->m_Metrics[ i ]->GetValue( parameters );
      measure += this->m_MetricValues[ i ];
    }

    /** Return a value. */
    return measure;

  } // end GetValue


  /**
	 * ********************* GetDerivative ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    CombinationImageToImageMetric<TFixedImage,TMovingImage>
    ::GetDerivative( const ParametersType & parameters,
      DerivativeType & derivative ) const
  {
    /** Add all metric derivatives. */
    DerivativeType tmpDerivative = DerivativeType( this->GetNumberOfParameters() );
    derivative = DerivativeType( this->GetNumberOfParameters() );
    derivative.Fill( NumericTraits< MeasureType >::Zero );
    for ( unsigned int i = 0; i < this->m_NumberOfMetrics; i++ )
    {
      tmpDerivative.Fill( NumericTraits< MeasureType >::Zero );
      this->m_Metrics[ i ]->GetDerivative( parameters, tmpDerivative );
      derivative += this->m_MetricWeights[ i ] * tmpDerivative;
    }

  } // end GetDerivative


  /**
	 * ********************* GetValueAndDerivative ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    CombinationImageToImageMetric<TFixedImage,TMovingImage>
    ::GetValueAndDerivative( const ParametersType & parameters,
      MeasureType & value, DerivativeType & derivative ) const
  {
    /** Initialise. */
    MeasureType tmpValue = NumericTraits< MeasureType >::Zero;
    DerivativeType tmpDerivative = DerivativeType( this->GetNumberOfParameters() );

    value = NumericTraits< MeasureType >::Zero;
    derivative = DerivativeType( this->GetNumberOfParameters() );
    derivative.Fill( NumericTraits< MeasureType >::Zero );

    /** Add all metric values and derivatives. */
    for ( unsigned int i = 0; i < this->m_NumberOfMetrics; i++ )
    {
      tmpValue = NumericTraits< MeasureType >::Zero;
      tmpDerivative.Fill( NumericTraits< MeasureType >::Zero );
      this->m_Metrics[ i ]->GetValueAndDerivative( parameters, tmpValue, tmpDerivative );
      this->m_MetricValues[ i ] = this->m_MetricWeights[ i ] * tmpValue;
      value += this->m_MetricValues[ i ];
      derivative += this->m_MetricWeights[ i ] * tmpDerivative;
    }

  } // end GetValueAndDerivative


  template <class TFixedImage, class TMovingImage>
  unsigned long
  CombinationImageToImageMetric<TFixedImage,TMovingImage>
  ::GetMTime() const
  {
    unsigned long mtime = this->Superclass::GetMTime();
    unsigned long m;
  
    // Some of the following should be removed once this 'ivars' are put in the
    // input and output lists

    /** Check the modified time of the sub metrics */  
    for (unsigned int i = 0; i < this->GetNumberOfMetrics(); ++i)
    {
      SingleValuedCostFunctionPointer metric = this->GetMetric(i);
      if ( metric.IsNotNull() )
      {
        m = metric->GetMTime();
        mtime = (m > mtime ? m : mtime);
      }
    }
  
    return mtime;  
  } // end GetMTime


} // end namespace itk

#undef itkImplementationSetObjectMacro
#undef itkImplementationSetConstObjectMacro
#undef itkImplementationGetConstObjectMacro

#endif // end #ifndef _itkCombinationImageToImageMetric_txx

