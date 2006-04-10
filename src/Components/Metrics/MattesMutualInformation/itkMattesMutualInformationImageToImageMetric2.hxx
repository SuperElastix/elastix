/**
 * This file is an adapted version of the original itk-class.
 *
 * The original itk-copyright message is stated below:
 */


/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile$
  Language:  C++
  Date:      $Date$
  Version:   $Revision$
	
		Copyright (c) Insight Software Consortium. All rights reserved.
		See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.
		
			This software is distributed WITHOUT ANY WARRANTY; without even 
			the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
			PURPOSE.  See the above copyright notices for more information.
			
=========================================================================*/
#ifndef _itkMattesMutualInformationImageToImageMetric2_HXX__
#define _itkMattesMutualInformationImageToImageMetric2_HXX__

#include "itkMattesMutualInformationImageToImageMetric2.h"

#include "itkBSplineInterpolateImageFunction.h"
#include "itkCovariantVector.h"

#include "itkImageRegionConstIterator.h"
#include "itkImageRegionIterator.h"
#include "itkImageRandomConstIteratorWithIndex.h"

#include "itkBSplineDeformableTransform.h"
#include "vnl/vnl_math.h"



namespace itk
{
	
	
	/**
	 * ********************* Constructor ****************************
	 */

	template < class TFixedImage, class TMovingImage >
		MattesMutualInformationImageToImageMetric2<TFixedImage,TMovingImage>
		::MattesMutualInformationImageToImageMetric2()
	{
		/** Initialize.*/
		this->m_NumberOfSpatialSamples = 500;
		this->m_NumberOfHistogramBins = 50;
		
		this->SetComputeGradient(false); // don't use the default gradient for now
		
		this->m_InterpolatorIsBSpline = false;
		this->m_TransformIsBSpline    = false;
		
		// Initialize PDFs to NULL
		this->m_JointPDF = NULL;
		this->m_JointPDFDerivatives = NULL;
		
		typename BSplineTransformType::Pointer transformer = BSplineTransformType::New();
		this->SetTransform (transformer);
		
		typename BSplineInterpolatorType::Pointer interpolator = BSplineInterpolatorType::New();
		this->SetInterpolator (interpolator);
		
		// Initialize memory
		this->m_MovingImageNormalizedMin = 0.0;
		this->m_FixedImageNormalizedMin = 0.0;
		this->m_MovingImageTrueMin = 0.0;
		this->m_MovingImageTrueMax = 0.0;
		this->m_FixedImageBinSize = 0.0;
		this->m_MovingImageBinSize = 0.0;
		this->m_CubicBSplineDerivativeKernel = NULL;
		this->m_BSplineInterpolator = NULL;
		this->m_DerivativeCalculator = NULL;
		this->m_NumParametersPerDim = 0;
		this->m_NumBSplineWeights = 0;
		this->m_BSplineTransform = NULL;
		this->m_NumberOfParameters = 0;
	
		// Added for elastix
		this->m_UseAllPixels = false;
		this->m_AllFixedImagePixelsStoredInContainer = false;
		
	} // end Constructor
	
	

	/**
	 * ********************* PrintSelf ******************************
	 *
	 * Print out internal information about this class.
	 */

	template < class TFixedImage, class TMovingImage  >
		void
		MattesMutualInformationImageToImageMetric2<TFixedImage,TMovingImage>
		::PrintSelf( std::ostream& os, Indent indent ) const
	{
		/** Call the superclass' PrintSelf. */
		Superclass::PrintSelf( os, indent );
		
		/** Add debugging information. */
		os << indent << "NumberOfSpatialSamples: ";
		os << this->m_NumberOfSpatialSamples << std::endl;
		os << indent << "NumberOfHistogramBins: ";
		os << this->m_NumberOfHistogramBins << std::endl;
	  os << indent << "UseAllPixels: ";
	  os << m_UseAllPixels << std::endl;

		os << indent << "NumberOfParameters: ";
		os << this->m_NumberOfParameters << std::endl;
		os << indent << "FixedImageNormalizedMin: ";
		os << this->m_FixedImageNormalizedMin << std::endl;
		os << indent << "MovingImageNormalizedMin: ";
		os << this->m_MovingImageNormalizedMin << std::endl;
		os << indent << "MovingImageTrueMin: ";
		os << this->m_MovingImageTrueMin << std::endl;
		os << indent << "MovingImageTrueMax: ";
		os << this->m_MovingImageTrueMax << std::endl;
		os << indent << "FixedImageBinSize: "; 
		os << this->m_FixedImageBinSize << std::endl;
		os << indent << "MovingImageBinSize: ";
		os << this->m_MovingImageBinSize << std::endl;
		os << indent << "InterpolatorIsBSpline: ";
		os << this->m_InterpolatorIsBSpline << std::endl;
		os << indent << "TransformIsBSpline: ";
		os << this->m_TransformIsBSpline << std::endl;
		
	} // end PrintSelf
	
	
	/**
	 * ********************* Initialize *****************************
	 */

	template <class TFixedImage, class TMovingImage> 
		void
		MattesMutualInformationImageToImageMetric2<TFixedImage,TMovingImage>
		::Initialize(void) throw ( ExceptionObject )
	{
		
		this->Superclass::Initialize();
		
		/** Cache the number of transformation parameters.*/
		this->m_NumberOfParameters = this->m_Transform->GetNumberOfParameters();
		
		/**
		 * Compute the minimum and maximum for the FixedImage over
		 * the FixedImageRegion.
		 *
		 * NB: We can't use StatisticsImageFilter to do this because
		 * the filter computes the min/max for the largest possible region.
		 */
		double fixedImageMin = NumericTraits<double>::max();
		double fixedImageMax = NumericTraits<double>::NonpositiveMin();
		
		typedef ImageRegionConstIterator<FixedImageType> FixedIteratorType;
		FixedIteratorType fixedImageIterator( 
			this->m_FixedImage, this->GetFixedImageRegion() );
		
		for ( fixedImageIterator.GoToBegin(); 
      		!fixedImageIterator.IsAtEnd(); ++fixedImageIterator )
    {
			
			double sample = static_cast<double>( fixedImageIterator.Get() );
			
			if ( sample < fixedImageMin )
      {
				fixedImageMin = sample;
      }
			
			if ( sample > fixedImageMax )
      {
				fixedImageMax = sample;
      }
    }
		
		/**
		 * Compute the minimum and maximum for the entire moving image
		 * in the buffer.
		 */
		double movingImageMin = NumericTraits<double>::max();
		double movingImageMax = NumericTraits<double>::NonpositiveMin();
		
		typedef ImageRegionConstIterator<MovingImageType> MovingIteratorType;
		MovingIteratorType movingImageIterator(
			this->m_MovingImage, this->m_MovingImage->GetBufferedRegion() );
		
		for ( movingImageIterator.GoToBegin(); 
		!movingImageIterator.IsAtEnd(); ++movingImageIterator)
    {
			double sample = static_cast<double>( movingImageIterator.Get() );
			
			if ( sample < movingImageMin )
      {
				movingImageMin = sample;
      }
			
			if ( sample > movingImageMax )
      {
				movingImageMax = sample;
      }
    }
		
		this->m_MovingImageTrueMin = movingImageMin;
		this->m_MovingImageTrueMax = movingImageMax;
		
		itkDebugMacro( " FixedImageMin: " << fixedImageMin << 
			" FixedImageMax: " << fixedImageMax << std::endl );
		itkDebugMacro( " MovingImageMin: " << movingImageMin << 
			" MovingImageMax: " << movingImageMax << std::endl );
		
		
		/**
		 * Compute binsize for the histograms.
		 *
		 * The binsize for the image intensities needs to be adjusted so that 
		 * we can avoid dealing with boundary conditions using the cubic 
		 * spline as the Parzen window.  We do this by increasing the size
		 * of the bins so that the joint histogram becomes "padded" at the 
		 * borders. Because we are changing the binsize, 
		 * we also need to shift the minimum by the padded amount in order to 
		 * avoid minimum values filling in our padded region.
		 *
		 * Note that there can still be non-zero bin values in the padded region,
		 * it's just that these bins will never be a central bin for the Parzen
		 * window.
		 */
		const int padding = 2;  // this will pad by 2 bins
		
		this->m_FixedImageBinSize = ( fixedImageMax - fixedImageMin ) /
			static_cast<double>( this->m_NumberOfHistogramBins - 2 * padding );
		this->m_FixedImageNormalizedMin = fixedImageMin / this->m_FixedImageBinSize - 
			static_cast<double>( padding );
		
		this->m_MovingImageBinSize = ( movingImageMax - movingImageMin ) /
			static_cast<double>( this->m_NumberOfHistogramBins - 2 * padding );
		this->m_MovingImageNormalizedMin = movingImageMin / this->m_MovingImageBinSize -
			static_cast<double>( padding );
		
		
		itkDebugMacro( "FixedImageNormalizedMin: " << this->m_FixedImageNormalizedMin );
		itkDebugMacro( "MovingImageNormalizedMin: " << this->m_MovingImageNormalizedMin );
		itkDebugMacro( "FixedImageBinSize: " << this->m_FixedImageBinSize );
		itkDebugMacro( "MovingImageBinSize; " << this->m_MovingImageBinSize );
		
		/**
		* Allocate memory for the fixed image sample containers.
		*/
		this->m_FixedImageSamples.resize( this->m_NumberOfSpatialSamples);
		this->m_AllFixedImagePixels.clear();

		/**
		* Allocate memory for the marginal PDF and initialize values
		* to zero. The marginal PDFs are stored as std::vector.
		*/
		this->m_FixedImageMarginalPDF.resize( this->m_NumberOfHistogramBins, 0.0 );
		this->m_MovingImageMarginalPDF.resize( this->m_NumberOfHistogramBins, 0.0 );
		
		/**
		* Allocate memory for the joint PDF and joint PDF derivatives.
		* The joint PDF and joint PDF derivatives are store as itk::Image.
		*/
		this->m_JointPDF = JointPDFType::New();
		this->m_JointPDFDerivatives = JointPDFDerivativesType::New();
		
		// Instantiate a region, index, size
		JointPDFRegionType            jointPDFRegion;
		JointPDFIndexType              jointPDFIndex;
		JointPDFSizeType              jointPDFSize;
		
		JointPDFDerivativesRegionType  jointPDFDerivativesRegion;
		JointPDFDerivativesIndexType  jointPDFDerivativesIndex;
		JointPDFDerivativesSizeType    jointPDFDerivativesSize;
		
		// For the joint PDF define a region starting from {0,0} 
		// with size {this->m_NumberOfHistogramBins, this->m_NumberOfHistogramBins}
		// The dimension represents fixed image parzen window index
	  // and moving image parzen window index, respectively.
		jointPDFIndex.Fill( 0 ); 
		jointPDFSize.Fill( this->m_NumberOfHistogramBins ); 
		
		jointPDFRegion.SetIndex( jointPDFIndex );
		jointPDFRegion.SetSize( jointPDFSize );
		
		// Set the regions and allocate
		this->m_JointPDF->SetRegions( jointPDFRegion );
		this->m_JointPDF->Allocate();
		
		// For the derivatives of the joint PDF define a region starting from {0,0,0} 
		// with size {m_NumberOfParameters,m_NumberOfHistogramBins, 
		// m_NumberOfHistogramBins}. The dimension represents transform parameters,
		// moving image parzen window index and fixed image parzen window index,
		// respectively. 
		jointPDFDerivativesIndex.Fill( 0 ); 
		jointPDFDerivativesSize[0] = this->m_NumberOfParameters;
		jointPDFDerivativesSize[1] = this->m_NumberOfHistogramBins;
		jointPDFDerivativesSize[2] = this->m_NumberOfHistogramBins;
		
		jointPDFDerivativesRegion.SetIndex( jointPDFDerivativesIndex );
		jointPDFDerivativesRegion.SetSize( jointPDFDerivativesSize );
		
		// Set the regions and allocate
		this->m_JointPDFDerivatives->SetRegions( jointPDFDerivativesRegion );
		this->m_JointPDFDerivatives->Allocate();
		
		
		/**
		* Setup the kernels used for the Parzen windows.
		*/
		this->m_CubicBSplineKernel = CubicBSplineFunctionType::New();
		this->m_CubicBSplineDerivativeKernel = CubicBSplineDerivativeFunctionType::New();    
		
		
		/** 
		* Uniformly sample the fixed image (within the fixed image region)
		* to create the sample points list.
		*/
		this->m_AllFixedImagePixelsStoredInContainer = false;
		if (this->GetUseAllPixels())
		{
			this->SampleFullFixedImageDomain( this->m_AllFixedImagePixels );
			this->SetNumberOfSpatialSamples( this->m_AllFixedImagePixels.size() );
		}
		else
		{
		  this->SampleFixedImageDomain( this->m_FixedImageSamples );
		}

		
		/**
		* Check if the interpolator is of type BSplineInterpolateImageFunction.
		* If so, we can make use of its EvaluateDerivatives method.
		* Otherwise, we instantiate an external central difference
		* derivative calculator.
		*/

		/** \todo Also add it the possibility of using the default gradient
		 * provided by the superclass.
		 */
		this->m_InterpolatorIsBSpline = true;
		
		BSplineInterpolatorType * testPtr = dynamic_cast<BSplineInterpolatorType *>(
			this->m_Interpolator.GetPointer() );
		if ( !testPtr )
    {
			this->m_InterpolatorIsBSpline = false;
			
			this->m_DerivativeCalculator = DerivativeFunctionType::New();
			this->m_DerivativeCalculator->SetInputImage( this->m_MovingImage );
			
			this->m_BSplineInterpolator = NULL;
			itkDebugMacro( "Interpolator is not BSpline" );
    } 
		else
    {
			this->m_BSplineInterpolator = testPtr;
			this->m_DerivativeCalculator = NULL;
			itkDebugMacro( "Interpolator is BSpline" );
    }
		
		/** 
		* Check if the transform is of type BSplineDeformableTransform.
		* If so, we can speed up derivative calculations by only inspecting
		* the parameters in the support region of a point.
		*
		*/
		this->m_TransformIsBSpline = true;
		
		BSplineTransformType * testPtr2 = dynamic_cast<BSplineTransformType *>(
			this->m_Transform.GetPointer() );
		if( !testPtr2 )
    {
			this->m_TransformIsBSpline = false;
			this->m_BSplineTransform = NULL;
			itkDebugMacro( "Transform is not BSplineDeformable" );
    }
		else
    {
			this->m_BSplineTransform = testPtr2;
			this->m_NumParametersPerDim = this->m_BSplineTransform->GetNumberOfParametersPerDimension();
			this->m_NumBSplineWeights = this->m_BSplineTransform->GetNumberOfWeights();
			itkDebugMacro( "Transform is BSplineDeformable" );
    }
		
		if ( this->m_TransformIsBSpline )
    {
			this->m_BSplineTransformWeights = BSplineTransformWeightsType( this->m_NumBSplineWeights );
			this->m_BSplineTransformIndices = BSplineTransformIndexArrayType( this->m_NumBSplineWeights );
			for ( unsigned int j = 0; j < FixedImageDimension; j++ )
      {
				this->m_ParametersOffset[j] = j * 
          this->m_BSplineTransform->GetNumberOfParametersPerDimension(); 
      }
    }
		
	} // end Initialize


	/**
	 * ******************* SampleFixedImageDomain *******************
	 *
	 * Uniformly sample the fixed image domain using a random walk.
	 */
	 
	 template < class TFixedImage, class TMovingImage >
		 void
		 MattesMutualInformationImageToImageMetric2<TFixedImage,TMovingImage>
		 ::SampleFixedImageDomain( FixedImageSpatialSampleContainer& samples )
	 {		 

		 /** Set up a random interator within the user specified fixed image region.*/
		 		 
		 typedef ImageRandomConstIteratorWithIndex<FixedImageType> RandomIterator;
		 RandomIterator randIter( this->m_FixedImage, this->GetFixedImageRegion() );
		 randIter.GoToBegin();
		 
		 typename FixedImageSpatialSampleContainer::iterator iter;
		 typename FixedImageSpatialSampleContainer::const_iterator end=samples.end();
		 
		 if (!(this->m_FixedImageMask))
		 {
			 randIter.SetNumberOfSamples( this->m_NumberOfSpatialSamples );
			 
			 for( iter=samples.begin(); iter != end; ++iter )
			 {
				 // Get sampled index
				 FixedImageIndexType index = randIter.GetIndex();
				 // Get sampled fixed image value
				 (*iter).FixedImageValue = randIter.Get();
				 // Translate index to point
				 this->m_FixedImage->TransformIndexToPhysicalPoint( index,
					 (*iter).FixedImagePointValue );
				 // Jump to random position
				 ++randIter;
			 }
		 }
		 else
		 {
			 FixedImagePointType inputPoint;
			 // No real meaning in this:
			 randIter.SetNumberOfSamples( this->m_NumberOfSpatialSamples );
			 
			 for( iter=samples.begin(); iter != end; ++iter )
			 {
				 //added for mask support.
				 do 
				 {
					 //Jump to random position
					 ++randIter;
					 
					 // Get sampled index
					 FixedImageIndexType index = randIter.GetIndex();
					 
					 // Translate index to point
					 this->m_FixedImage->TransformIndexToPhysicalPoint( index,
						 inputPoint );
					 
				 } while ( !(this->m_FixedImageMask->IsInside(inputPoint)) );
				 
				 // Get sampled fixed image value and position
				 (*iter).FixedImagePointValue = inputPoint;
				 (*iter).FixedImageValue = randIter.Get();
				 
			 } // end for iter=samples...
			 
		 } // end else (if mask exists)
		 
	 } // end SampleFixedImageDomain
	 

	/**
	 * ******************* SampleFullFixedImageDomain *********************
	 *
	 * Store all pixels in the fixed image domain in a container
	 */
	 
	 template < class TFixedImage, class TMovingImage >
		 void
		 MattesMutualInformationImageToImageMetric2<TFixedImage,TMovingImage>
		 ::SampleFullFixedImageDomain( FixedImageSpatialSampleContainer& samples )
	 {		 

		 samples.clear();

		 /** Set up a region interator within the user specified fixed image region.*/
	   typedef ImageRegionConstIteratorWithIndex<FixedImageType> FixedImageIterator;
		 FixedImageIterator fiter( this->m_FixedImage, this->GetFixedImageRegion() );
			 
		 if (!(this->m_FixedImageMask))
		 {
			 		 
			 for( fiter.GoToBegin(); ! fiter.IsAtEnd(); ++fiter )
			 {
				 FixedImageSpatialSample tempsample;

				 // Get sampled index
				 FixedImageIndexType index = fiter.GetIndex();

				 // Translate index to point
				 this->m_FixedImage->TransformIndexToPhysicalPoint( index,
					 tempsample.FixedImagePointValue );

				 // Get sampled fixed image value
				 tempsample.FixedImageValue = fiter.Get();

				 // Store in container
				 samples.push_back(tempsample);
			 }

		 }
		 else
		 {
			
			 for( fiter.GoToBegin(); ! fiter.IsAtEnd(); ++fiter )
			 {
				 FixedImageSpatialSample tempsample;

				 // Get sampled index
				 FixedImageIndexType index = fiter.GetIndex();
				 
				 // Translate index to point
				 this->m_FixedImage->TransformIndexToPhysicalPoint( index,
					 tempsample.FixedImagePointValue );

				 if ( this->m_FixedImageMask->IsInside( tempsample.FixedImagePointValue ) )
				 {
 				   // Get sampled fixed image value
	 			   tempsample.FixedImageValue = fiter.Get();

				   // Store in container
				   samples.push_back(tempsample);
				 }
			 }
			 
		 } // end else (if mask exists)

		 this->m_AllFixedImagePixelsStoredInContainer = true;
		 
	 } // end SampleFullFixedImageDomain
	 

	 
	/**
	 * ************************** GetValue **************************
	 *
	 * Get the match Measure.
	 */

	 template < class TFixedImage, class TMovingImage  >
		 typename MattesMutualInformationImageToImageMetric2<TFixedImage,TMovingImage>
		 ::MeasureType
		 MattesMutualInformationImageToImageMetric2<TFixedImage,TMovingImage>
		 ::GetValue( const ParametersType& parameters ) const
	 {		 
		 // Reset marginal pdf to all zeros.
		 // Assumed the size has already been set to NumberOfHistogramBins
		 // in Initialize().
		 for ( unsigned int j = 0; j < this->m_NumberOfHistogramBins; j++ )
		 {
			 this->m_FixedImageMarginalPDF[j]  = 0.0;
			 this->m_MovingImageMarginalPDF[j] = 0.0;
		 }
		 
		 // Reset the joint pdfs to zero
		 this->m_JointPDF->FillBuffer( 0.0 );
		 
		 // Set up the parameters in the transform
		 this->m_Transform->SetParameters( parameters );
		 
		 // Declare iterators for iteration over the sample container
		 typename FixedImageSpatialSampleContainer::const_iterator fiter;
		 typename FixedImageSpatialSampleContainer::const_iterator fbegin;
		 typename FixedImageSpatialSampleContainer::const_iterator fend;

		 // Decide which sample container to use */
     if ( this->GetUseAllPixels() )
		 {
       fbegin = this->m_AllFixedImagePixels.begin();
		   fend = this->m_AllFixedImagePixels.end();
		 }
		 else
		 {
			 fbegin = this->m_FixedImageSamples.begin();
		   fend = this->m_FixedImageSamples.end();
		 }

		 
		 unsigned long nSamples=0;
		 unsigned long nFixedImageSamples=0;
		 		 
		 for ( fiter = fbegin; fiter != fend; ++fiter )
		 {			 
			 ++nFixedImageSamples;
			 
			 // Get moving image value
			 MovingImagePointType mappedPoint;
			 bool sampleOk;
			 double movingImageValue;
			 
			 this->TransformPoint( (*fiter).FixedImagePointValue, mappedPoint, 
				 sampleOk, movingImageValue );
			 
			 if( sampleOk )
			 {
				 
				 ++nSamples; 
				 
				 /**
				 * Compute this sample's contribution to the marginal and joint distributions.
				 *
				 */
				 
				 // Determine parzen window arguments (see eqn 6 of Mattes paper [2]).    
				 double movingImageParzenWindowTerm =
					 movingImageValue / this->m_MovingImageBinSize - this->m_MovingImageNormalizedMin;
				 unsigned int movingImageParzenWindowIndex = 
					 static_cast<unsigned int>( vcl_floor( movingImageParzenWindowTerm ) );
				 
				 double fixedImageParzenWindowTerm = 
					 static_cast<double>( (*fiter).FixedImageValue ) / this->m_FixedImageBinSize -
					 this->m_FixedImageNormalizedMin;
				 unsigned int fixedImageParzenWindowIndex =
					 static_cast<unsigned int>( vcl_floor( fixedImageParzenWindowTerm ) );
				 
				 
				 // Make sure the extreme values are in valid bins
				 if ( fixedImageParzenWindowIndex < 2 )
				 {
					 fixedImageParzenWindowIndex = 2;
				 }
				 else if ( fixedImageParzenWindowIndex > (this->m_NumberOfHistogramBins - 3) )
				 {
					 fixedImageParzenWindowIndex = this->m_NumberOfHistogramBins - 3;
				 }
				 
				 if ( movingImageParzenWindowIndex < 2 )
				 {
					 movingImageParzenWindowIndex = 2;
				 }
				 else if ( movingImageParzenWindowIndex > (this->m_NumberOfHistogramBins - 3) )
				 {
					 movingImageParzenWindowIndex = this->m_NumberOfHistogramBins - 3;
				 }
				 
				 
				 // Since a zero-order BSpline (box car) kernel is used for
				 // the fixed image marginal pdf, we need only increment the
				 // fixedImageParzenWindowIndex by value of 1.0.
				 this->m_FixedImageMarginalPDF[fixedImageParzenWindowIndex] +=
					 static_cast<PDFValueType>( 1 );
				 
				 /**
				  * The region of support of the parzen window determines which bins
				  * of the joint PDF are effected by the pair of image values.
				  * Since we are using a cubic spline for the moving image parzen
				  * window, four bins are effected.  The fixed image parzen window is
				  * a zero-order spline (box car) and thus effects only one bin.
				  *
				  *  The PDF is arranged so that moving image bins corresponds to the 
				  * zero-th (column) dimension and the fixed image bins corresponds
				  * to the first (row) dimension.
				  *
				  */

				 // Pointer to affected bin to be updated
		     JointPDFValueType *pdfPtr = this->m_JointPDF->GetBufferPointer() +
			     ( fixedImageParzenWindowIndex * this->m_JointPDF->GetOffsetTable()[1] );
 
				 // Move the pointer to the first affected bin
				 int pdfMovingIndex = static_cast<int>( movingImageParzenWindowIndex ) - 1;
				 pdfPtr += pdfMovingIndex;

				 for ( ; pdfMovingIndex <= static_cast<int>( movingImageParzenWindowIndex ) + 2;
						   pdfMovingIndex++, pdfPtr++ )
         {

					 // Update PDF for the current intensity pair
					 double movingImageParzenWindowArg = 
						 static_cast<double>( pdfMovingIndex ) - 
						 static_cast<double>( movingImageParzenWindowTerm );

					 *(pdfPtr) += static_cast<PDFValueType>( 
						 this->m_CubicBSplineKernel->Evaluate( movingImageParzenWindowArg ) );

				 }  //end parzen windowing for loop
			 
			 } //end if-block check sampleOk
			 
		 } // end iterating over fixed image spatial sample container for loop
		 
		 itkDebugMacro( "Ratio of voxels mapping into moving image buffer: " 
			 << nSamples << " / " << this->m_NumberOfSpatialSamples << std::endl );
		 
		 if( nSamples < this->m_NumberOfSpatialSamples / 4 )
		 {
			 itkExceptionMacro( "Too many samples map outside moving image buffer: "
				 << nSamples << " / " << this->m_NumberOfSpatialSamples << std::endl );
		 }	

		 this->m_NumberOfPixelsCounted = nSamples;
		 
		 /**
		 * Normalize the PDFs, compute moving image marginal PDF
		 *
		 */
		 typedef ImageRegionIterator<JointPDFType> JointPDFIteratorType;
		 JointPDFIteratorType jointPDFIterator ( this->m_JointPDF, this->m_JointPDF->GetBufferedRegion() );
		 
		 jointPDFIterator.GoToBegin();
		 
		 
		 // Compute joint PDF normalization factor (to ensure joint PDF sum adds to 1.0)
		 double jointPDFSum = 0.0;
		 
		 while( !jointPDFIterator.IsAtEnd() )
		 {
			 jointPDFSum += jointPDFIterator.Get();
			 ++jointPDFIterator;
		 }
		 
		 if ( jointPDFSum == 0.0 )
		 {
			 itkExceptionMacro( "Joint PDF summed to zero" );
		 }
		 
		 
		 // Normalize the PDF bins
		 jointPDFIterator.GoToEnd();
		 while( !jointPDFIterator.IsAtBegin() )
		 {
			 --jointPDFIterator;
			 jointPDFIterator.Value() /= static_cast<PDFValueType>( jointPDFSum );
		 }
		 
		 
		 // Normalize the fixed image marginal PDF
		 double fixedPDFSum = 0.0;
		 for( unsigned int bin = 0; bin < this->m_NumberOfHistogramBins; bin++ )
		 {
			 fixedPDFSum += this->m_FixedImageMarginalPDF[bin];
		 }
		 
		 if ( fixedPDFSum == 0.0 )
		 {
			 itkExceptionMacro( "Fixed image marginal PDF summed to zero" );
		 }
		 
		 for( unsigned int bin=0; bin < this->m_NumberOfHistogramBins; bin++ )
		 {
			 this->m_FixedImageMarginalPDF[bin] /= static_cast<PDFValueType>( fixedPDFSum );
		 }
		 
		 
		 // Compute moving image marginal PDF by summing over fixed image bins.
		 typedef ImageLinearIteratorWithIndex<JointPDFType> JointPDFLinearIterator;
		 JointPDFLinearIterator linearIter( 
			 this->m_JointPDF, this->m_JointPDF->GetBufferedRegion() );
		 
		 linearIter.SetDirection( 1 );
		 linearIter.GoToBegin();
		 unsigned int movingIndex = 0;
		 
		 while( !linearIter.IsAtEnd() )
		 {
			 
			 double sum = 0.0;
			 
			 while( !linearIter.IsAtEndOfLine() )
			 {
				 sum += linearIter.Get();
				 ++linearIter;
			 }
			 
			 this->m_MovingImageMarginalPDF[movingIndex] = static_cast<PDFValueType>(sum);
			 
			 linearIter.NextLine();
			 ++movingIndex;			 
		 }
		 
		 /** Compute the metric by double summation over histogram. */

		 // Setup pointer to point to the first bin
		 JointPDFValueType * jointPDFPtr = this->m_JointPDF->GetBufferPointer();

		 // Initialze sum to zero
		 double sum = 0.0;

		 for( unsigned int fixedIndex = 0; fixedIndex < this->m_NumberOfHistogramBins; ++fixedIndex )
		 {
			 double fixedImagePDFValue = this->m_FixedImageMarginalPDF[fixedIndex];

			 for( unsigned int movingIndex = 0; movingIndex < this->m_NumberOfHistogramBins;
				 ++movingIndex, jointPDFPtr++ )      
			 {
				 double movingImagePDFValue = this->m_MovingImageMarginalPDF[movingIndex];
				 double jointPDFValue = *(jointPDFPtr);
				 
				 // check for non-zero bin contribution
				 if( jointPDFValue > 1e-16 &&  movingImagePDFValue > 1e-16 )
				 {
					 double pRatio = vcl_log( jointPDFValue / movingImagePDFValue );
					 if( fixedImagePDFValue > 1e-16)
					 {
						 sum += jointPDFValue * ( pRatio - vcl_log( fixedImagePDFValue ) );
					 }
					 
				 }  // end if-block to check non-zero bin contribution
			 }  // end for-loop over moving index
		 }  // end for-loop over fixed index
		 
		 return static_cast<MeasureType>( -1.0 * sum );
		 
	} // end GetValue


	/**
	 * ******************** GetValueAndDerivative *******************
	 *
	 * Get both the Value and the Derivative of the Measure. 
	 * Both are computed on a randomly chosen set of voxels in the
	 * fixed image domain or on all pixels.
	 */

	 template < class TFixedImage, class TMovingImage  >
		 void
		 MattesMutualInformationImageToImageMetric2<TFixedImage,TMovingImage>
		 ::GetValueAndDerivative(
		 const ParametersType& parameters,
		 MeasureType& value,
		 DerivativeType& derivative) const
	 {		 
		 // Set output values to zero
		 value = NumericTraits< MeasureType >::Zero;
		 derivative = DerivativeType( this->GetNumberOfParameters() );
		 derivative.Fill( NumericTraits< MeasureType >::Zero );
		 
		 
		 // Reset marginal pdf to all zeros.
		 // Assumed the size has already been set to NumberOfHistogramBins
		 // in Initialize().
		 for ( unsigned int j = 0; j < this->m_NumberOfHistogramBins; j++ )
		 {
			 this->m_FixedImageMarginalPDF[j]  = 0.0;
			 this->m_MovingImageMarginalPDF[j] = 0.0;
		 }
		 
		 // Reset the joint pdfs to zero
		 this->m_JointPDF->FillBuffer( 0.0 );
		 this->m_JointPDFDerivatives->FillBuffer( 0.0 );
		 
		 
		 // Set up the parameters in the transform
		 this->m_Transform->SetParameters( parameters );
		 
		 
		 // Declare iterators for iteration over the sample container
		 typename FixedImageSpatialSampleContainer::const_iterator fiter;
		 typename FixedImageSpatialSampleContainer::const_iterator fbegin;
		 typename FixedImageSpatialSampleContainer::const_iterator fend;

		 // Decide which sample container to use */
     if ( this->GetUseAllPixels() )
		 {
       fbegin = this->m_AllFixedImagePixels.begin();
		   fend = this->m_AllFixedImagePixels.end();
		 }
		 else
		 {
			 fbegin = this->m_FixedImageSamples.begin();
		   fend = this->m_FixedImageSamples.end();
		 }


		 unsigned long nSamples=0;
		 unsigned long nFixedImageSamples=0;
		 
		 for ( fiter = fbegin; fiter != fend; ++fiter )
		 {
			 
			 ++nFixedImageSamples;
			 
			 // Get moving image value
			 MovingImagePointType mappedPoint;
			 bool sampleOk;
			 double movingImageValue;
			 
			 this->TransformPoint( (*fiter).FixedImagePointValue, mappedPoint, 
				 sampleOk, movingImageValue );
			 
			 
			 if( sampleOk )
			 {
				 
				 ++nSamples; 
				 
				 ImageDerivativesType movingImageGradientValue;
				 this->ComputeImageDerivatives( mappedPoint, movingImageGradientValue );
				 
				 
				 /**
				 * Compute this sample's contribution to the marginal and joint distributions.
				 *
				 */
				 
				 // Determine parzen window arguments (see eqn 6 of Mattes paper [2]).    
				 double movingImageParzenWindowTerm =
					 movingImageValue / this->m_MovingImageBinSize - this->m_MovingImageNormalizedMin;
				 unsigned int movingImageParzenWindowIndex = 
					 static_cast<unsigned int>( vcl_floor( movingImageParzenWindowTerm ) );
				 
				 double fixedImageParzenWindowTerm = 
					 static_cast<double>( (*fiter).FixedImageValue ) / this->m_FixedImageBinSize -
					 this->m_FixedImageNormalizedMin;
				 unsigned int fixedImageParzenWindowIndex =
					 static_cast<unsigned int>( vcl_floor( fixedImageParzenWindowTerm ) );
				 				 
				 // Make sure the extreme values are in valid bins
				 if ( fixedImageParzenWindowIndex < 2 )
				 {
					 fixedImageParzenWindowIndex = 2;
				 }
				 else if ( fixedImageParzenWindowIndex > (this->m_NumberOfHistogramBins - 3) )
				 {
					 fixedImageParzenWindowIndex = this->m_NumberOfHistogramBins - 3;
				 }
				 
				 if ( movingImageParzenWindowIndex < 2 )
				 {
					 movingImageParzenWindowIndex = 2;
				 }
				 else if ( movingImageParzenWindowIndex > (this->m_NumberOfHistogramBins - 3) )
				 {
					 movingImageParzenWindowIndex = this->m_NumberOfHistogramBins - 3;
				 }
				 
				 
				 // Since a zero-order BSpline (box car) kernel is used for
				 // the fixed image marginal pdf, we need only increment the
				 // fixedImageParzenWindowIndex by value of 1.0.
				 this->m_FixedImageMarginalPDF[fixedImageParzenWindowIndex] += 
					 static_cast<PDFValueType>( 1 );
				 
				 /**
				  * The region of support of the parzen window determines which bins
				  * of the joint PDF are effected by the pair of image values.
				  * Since we are using a cubic spline for the moving image parzen
				  * window, four bins are effected.  The fixed image parzen window is
				  * a zero-order spline (box car) and thus effects only one bin.
				  *
				  *  The PDF is arranged so that moving image bins corresponds to the 
				  * zero-th (column) dimension and the fixed image bins corresponds
				  * to the first (row) dimension.
				  *
				  */

				 // Pointer to affected bin to be updated
				 JointPDFValueType *pdfPtr = this->m_JointPDF->GetBufferPointer() +
				   ( fixedImageParzenWindowIndex * m_NumberOfHistogramBins );
 
				 // Move the pointer to the fist affected bin
				 int pdfMovingIndex = static_cast<int>( movingImageParzenWindowIndex ) - 1;
				 pdfPtr += pdfMovingIndex;

				 for ( ; pdfMovingIndex <= static_cast<int>( movingImageParzenWindowIndex ) + 2;
            pdfMovingIndex++, pdfPtr++ )
         {

           // Update PDF for the current intensity pair
					 double movingImageParzenWindowArg = 
						 static_cast<double>( pdfMovingIndex ) - 
						 static_cast<double>( movingImageParzenWindowTerm );

					 *(pdfPtr) += static_cast<PDFValueType>( 
						 this->m_CubicBSplineKernel->Evaluate( movingImageParzenWindowArg ) );
	  
					 // Compute the cubicBSplineDerivative for later repeated use.
						double cubicBSplineDerivativeValue = 
							this->m_CubicBSplineDerivativeKernel->Evaluate( movingImageParzenWindowArg );
						 
					 // Compute PDF derivative contribution.
		 		 	 this->ComputePDFDerivatives( (*fiter).FixedImagePointValue, fixedImageParzenWindowIndex,
						 pdfMovingIndex, movingImageGradientValue, cubicBSplineDerivativeValue );
										 
				 }  //end parzen windowing for loop
				 
			 } //end if-block check sampleOk
			 
    } // end iterating over fixed image spatial sample container for loop
		
		itkDebugMacro( "Ratio of voxels mapping into moving image buffer: " 
			<< nSamples << " / " << this->m_NumberOfSpatialSamples << std::endl );
		
		if( nSamples < this->m_NumberOfSpatialSamples / 4 )
    {
			itkExceptionMacro( "Too many samples map outside moving image buffer: "
				<< nSamples << " / " << this->m_NumberOfSpatialSamples << std::endl );
    }
	  
		this->m_NumberOfPixelsCounted = nSamples;
						
		/**
		* Normalize the PDFs, compute moving image marginal PDF
		*
		*/
		typedef ImageRegionIterator<JointPDFType> JointPDFIteratorType;
		JointPDFIteratorType jointPDFIterator ( this->m_JointPDF, this->m_JointPDF->GetBufferedRegion() );
		
		jointPDFIterator.GoToBegin();
		
		
		// Compute joint PDF normalization factor (to ensure joint PDF sum adds to 1.0)
		double jointPDFSum = 0.0;
		
		while( !jointPDFIterator.IsAtEnd() )
    {
			jointPDFSum += jointPDFIterator.Get();
			++jointPDFIterator;
    }
		
		if ( jointPDFSum == 0.0 )
    {
			itkExceptionMacro( "Joint PDF summed to zero" );
    }
		
		
		// Normalize the PDF bins
		jointPDFIterator.GoToEnd();
		while( !jointPDFIterator.IsAtBegin() )
    {
			--jointPDFIterator;
			jointPDFIterator.Value() /= static_cast<PDFValueType>( jointPDFSum );
    }
		
		
		// Normalize the fixed image marginal PDF
		double fixedPDFSum = 0.0;
		for( unsigned int bin = 0; bin < this->m_NumberOfHistogramBins; bin++ )
    {
			fixedPDFSum += this->m_FixedImageMarginalPDF[bin];
    }
		
		if ( fixedPDFSum == 0.0 )
    {
			itkExceptionMacro( "Fixed image marginal PDF summed to zero" );
    }
		
		for( unsigned int bin=0; bin < this->m_NumberOfHistogramBins; bin++ )
    {
			this->m_FixedImageMarginalPDF[bin] /= static_cast<PDFValueType>( fixedPDFSum );
    }
		
		
		// Compute moving image marginal PDF by summing over fixed image bins.
		typedef ImageLinearIteratorWithIndex<JointPDFType> JointPDFLinearIterator;
		JointPDFLinearIterator linearIter( 
			this->m_JointPDF, this->m_JointPDF->GetBufferedRegion() );
		
		linearIter.SetDirection( 1 );
		linearIter.GoToBegin();
		unsigned int movingIndex = 0;
		
		while( !linearIter.IsAtEnd() )
    {
			
			double sum = 0.0;
			
			while( !linearIter.IsAtEndOfLine() )
      {
				sum += linearIter.Get();
				++linearIter;
      }
			
			this->m_MovingImageMarginalPDF[movingIndex] = static_cast<PDFValueType>(sum);
			
			linearIter.NextLine();
			++movingIndex;
			
    }
		
		
		// Normalize the joint PDF derivatives by the test image binsize and nSamples
		typedef ImageRegionIterator<JointPDFDerivativesType> JointPDFDerivativesIteratorType;
		JointPDFDerivativesIteratorType jointPDFDerivativesIterator (
			this->m_JointPDFDerivatives, this->m_JointPDFDerivatives->GetBufferedRegion() );
		
		jointPDFDerivativesIterator.GoToBegin();
		
		double nFactor = 1.0 / ( this->m_MovingImageBinSize * static_cast<double>( nSamples ) );
		
		while( !jointPDFDerivativesIterator.IsAtEnd() )
    {
			jointPDFDerivativesIterator.Value() *= nFactor;
			++jointPDFDerivativesIterator;
    }
		
		
		/** Compute the metric by double summation over histogram. */

		// Setup pointer to point to the first bin
		JointPDFValueType * jointPDFPtr = m_JointPDF->GetBufferPointer();

	  // Initialize sum to zero
		double sum = 0.0;

		for( unsigned int fixedIndex = 0; fixedIndex < this->m_NumberOfHistogramBins; ++fixedIndex )
    {
			double fixedImagePDFValue = this->m_FixedImageMarginalPDF[fixedIndex];

			for( unsigned int movingIndex = 0; movingIndex < this->m_NumberOfHistogramBins; 
				++movingIndex, jointPDFPtr++ )      
      {
				double movingImagePDFValue = this->m_MovingImageMarginalPDF[movingIndex];
				double jointPDFValue = *(jointPDFPtr);
				
				// check for non-zero bin contribution
				if( jointPDFValue > 1e-16 &&  movingImagePDFValue > 1e-16 )
        {
					
					double pRatio = vcl_log( jointPDFValue / movingImagePDFValue );
					if( fixedImagePDFValue > 1e-16)
					{
						sum += jointPDFValue * ( pRatio - vcl_log( fixedImagePDFValue ) );
					}

					// move joint pdf derivative pointer to the right position
		      JointPDFValueType * derivPtr = this->m_JointPDFDerivatives->GetBufferPointer() +
				    ( fixedIndex * this->m_JointPDFDerivatives->GetOffsetTable()[2] ) +
				    ( movingIndex * this->m_JointPDFDerivatives->GetOffsetTable()[1] );

					for( unsigned int parameter=0; parameter < this->m_NumberOfParameters;
						  ++parameter, derivPtr++ )
          {
										
						// Ref: eqn 23 of Thevenaz & Unser paper [3]
						derivative[parameter] -= (*derivPtr) * pRatio;
						
          }  // end for-loop over parameters
        }  // end if-block to check non-zero bin contribution
      }  // end for-loop over moving index
    }  // end for-loop over fixed index
		
		value = static_cast<MeasureType>( -1.0 * sum );
		
	} // end GetValueAndDerivative


	/**
	 * ******************** GetDerivative ***************************
	 *
	 * Get the match measure derivative.
	 */

	template < class TFixedImage, class TMovingImage  >
	void
	MattesMutualInformationImageToImageMetric2<TFixedImage,TMovingImage>
	::GetDerivative( const ParametersType& parameters, DerivativeType & derivative ) const
	{
		MeasureType value;
		// call the combined version
		this->GetValueAndDerivative( parameters, value, derivative );

	} // end GetDerivative


	/**
	 * ****************** ComputeImageDerivatives *******************
	 *
	 * Compute image derivatives using a central difference function
	 * if we are not using a BSplineInterpolator, which includes
	 * derivatives.
	 */

	template < class TFixedImage, class TMovingImage >
		void
		MattesMutualInformationImageToImageMetric2<TFixedImage,TMovingImage>
		::ComputeImageDerivatives( 
		const MovingImagePointType& mappedPoint, 
		ImageDerivativesType& gradient ) const
	{		
		if( this->m_InterpolatorIsBSpline )
		{
			// Computed moving image gradient using derivative BSpline kernel.
			gradient = this->m_BSplineInterpolator->EvaluateDerivative( mappedPoint );
		}
		else
		{
			// For all generic interpolator use central differencing.
			gradient = this->m_DerivativeCalculator->Evaluate( mappedPoint );
		}
		
	} // end ComputeImageDerivatives
	

	/**
	 * ********************** TransformPoint ************************
	 *
	 * Transform a point from FixedImage domain to MovingImage domain.
	 * This function also checks if mapped point is within support region.
	 */

	template < class TFixedImage, class TMovingImage >
		void
		MattesMutualInformationImageToImageMetric2<TFixedImage,TMovingImage>
		::TransformPoint( 
		const FixedImagePointType& fixedImagePoint, 
		MovingImagePointType& mappedPoint,
		bool& sampleOk,
		double& movingImageValue ) const
	{
		
		bool insideBSValidRegion;
		
		if ( !(this->m_TransformIsBSpline) )
		{
			mappedPoint = this->m_Transform->TransformPoint( fixedImagePoint );
		}
		else
		{
			this->m_BSplineTransform->TransformPoint( fixedImagePoint,
				mappedPoint,
				this->m_BSplineTransformWeights,
				this->m_BSplineTransformIndices,
				insideBSValidRegion );
		}
		
		// Check if mapped point inside image buffer
		sampleOk = this->m_Interpolator->IsInsideBuffer( mappedPoint );
		
		if ( this->m_TransformIsBSpline )
		{
			// Check if mapped point is within the support region of a grid point.
			// This is neccessary for computing the metric gradient
			sampleOk = sampleOk && insideBSValidRegion;
		}
		
		//Added for support of masks:
		//if mask exists and point is inside buffer
		if ( this->m_MovingImageMask && sampleOk ) 
		{
			sampleOk = (this->m_MovingImageMask->IsInside(mappedPoint));
		}
		
		if ( sampleOk )
		{
			movingImageValue = this->m_Interpolator->Evaluate( mappedPoint );
			
			if ( movingImageValue < this->m_MovingImageTrueMin || 
				movingImageValue > this->m_MovingImageTrueMax )
			{
				// need to throw out this sample as it will not fall into a valid bin
				sampleOk = false;
			}
		}

	} // end TransformPoint
	
	
	/**
	 * ******************** ComputePDFDerivatives *******************
	 *
	 * Compute PDF derivatives contribution for each parameter.
	 */

	template < class TFixedImage, class TMovingImage >
		void
		MattesMutualInformationImageToImageMetric2<TFixedImage,TMovingImage>
		::ComputePDFDerivatives( 
		const FixedImagePointType& fixedImagePoint, 
		int fixedImageParzenWindowIndex,
		int pdfMovingIndex,
		const ImageDerivativesType& movingImageGradientValue,
		double cubicBSplineDerivativeValue ) const
	{
		// Update bins in the PDF derivatives for the current intensity pair
		JointPDFValueType * derivPtr = this->m_JointPDFDerivatives->GetBufferPointer() +
			( fixedImageParzenWindowIndex * this->m_JointPDFDerivatives->GetOffsetTable()[2] ) +
			( pdfMovingIndex * this->m_JointPDFDerivatives->GetOffsetTable()[1] );
		
		if( !(this->m_TransformIsBSpline) )
		{
			
			/**
			 * Generic version which works for all transforms.
			 */
			
			// Compute the transform Jacobian.
			typedef typename TransformType::JacobianType JacobianType;
			const JacobianType& jacobian = 
				this->m_Transform->GetJacobian( fixedImagePoint );
			
			for ( unsigned int mu = 0; mu < m_NumberOfParameters; mu++, derivPtr++ )
			{
				double innerProduct = 0.0;
				for ( unsigned int dim = 0; dim < FixedImageDimension; dim++ )
				{
					innerProduct += jacobian[dim][mu] * 
						movingImageGradientValue[dim];
				} //end dim loop
				
				*(derivPtr) -= innerProduct * cubicBSplineDerivativeValue;
			} //end mu loop
			
		}
		else
		{
			
			/**
			 * If the transform is of type BSplineDeformableTransform,
			 * we can obtain a speed up by only processing the affected parameters.
			 */
			for( unsigned int dim = 0; dim < FixedImageDimension; dim++ )
			{
				
				for( unsigned int mu = 0; mu < this->m_NumBSplineWeights; mu++ )
				{
					
					/* The array weights contains the Jacobian values in a 1-D array 
					 * (because for each parameter the Jacobian is non-zero in only 1 of the
					 * possible dimensions) which is multiplied by the moving image gradient. */
					double innerProduct = movingImageGradientValue[dim] * this->m_BSplineTransformWeights[mu];
										
					JointPDFValueType * ptr = 
						derivPtr
						+	this->m_BSplineTransformIndices[mu]
						+	this->m_ParametersOffset[dim];
					*(ptr) -= innerProduct * cubicBSplineDerivativeValue;
					
				} //end mu for loop
			} //end dim for loop
			
		} // end if-block transform is BSpline
		
	} // end ComputePDFDerivatives
	

	/**
	 * ************************** GetExactValue **************************
	 *
	 * Get the match Measure, computed on all voxels.
	 * This method is meant for users that want to do optimisation using
	 * just a subset of the samples, but want to know the exact value
	 * of the metric.
	 */

	 template < class TFixedImage, class TMovingImage  >
		 typename MattesMutualInformationImageToImageMetric2<TFixedImage,TMovingImage>
		 ::MeasureType
		 MattesMutualInformationImageToImageMetric2<TFixedImage,TMovingImage>
		 ::GetExactValue( const ParametersType& parameters )
	 {		 

		 /** The AllFixedImagePixelsContainer may not be initialised */
		 if ( !(this->m_AllFixedImagePixelsStoredInContainer) )
		 {
			 this->SampleFullFixedImageDomain(
				 this->m_AllFixedImagePixels );
		 }

		 /** Remember the state of the m_UseAllPixels flag and the number of spatial samples */
		 bool useAllPixelsBackup = this->GetUseAllPixels();
		 unsigned long numberOfSpatialSamplesBackup = this->GetNumberOfSpatialSamples();

		 /** Compute the value using all voxels */
		 this->SetUseAllPixels(true);
		 this->SetNumberOfSpatialSamples( this->m_AllFixedImagePixels.size() );
		 double returnvalue = this->GetValue(parameters);

		 /** Restore the the UseAllPixels flag and the number of spatial samples */
		 this->SetUseAllPixels(useAllPixelsBackup);
		 this->SetNumberOfSpatialSamples(numberOfSpatialSamplesBackup);

		 /** Return the metric value */
		 return returnvalue;
			 
	 } // end GetExactValue


	 /**
	  * *********************** ReinitializeSeed ********************
	  *
	  * Method to reinitialize the seed of the random number generator
	  */

	 template < class TFixedImage, class TMovingImage  > void
	 MattesMutualInformationImageToImageMetric2<TFixedImage,TMovingImage>
	 ::ReinitializeSeed()
	 {
		 Statistics::MersenneTwisterRandomVariateGenerator::GetInstance()->SetSeed();
	 } //end ReinitializeSeed


	 /**
	  * *********************** ReinitializeSeed ********************
	  *
	  * Method to reinitialize the seed of the random number generator
	  */

	 template < class TFixedImage, class TMovingImage  > void
	 MattesMutualInformationImageToImageMetric2<TFixedImage,TMovingImage>
	 ::ReinitializeSeed(int seed)
	 {
	   Statistics::MersenneTwisterRandomVariateGenerator::GetInstance()->SetSeed(seed);
	 } //end ReinitializeSeed




} // end namespace itk


#endif // end #ifndef _itkMattesMutualInformationImageToImageMetric2_HXX__

