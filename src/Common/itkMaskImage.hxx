#ifndef _itkMaskImage_cxx
#define _itkMaskImage_cxx

#include "itkMaskImage.h"

namespace itk
{
	

	/**
	 * ********************* Constructor ****************************
	 */

	template <class MaskPixelType,unsigned int MaskDimension, class CoordType >
		MaskImage<MaskPixelType,MaskDimension,CoordType>
		::MaskImage()
		: Superclass() 
	{
		/** Initialize.*/
		this->m_Interpolator = MaskInterpolator::New();
		this->m_Interpolator->SetInputImage(this);
		
	}	// end Constructor
	
	/**
	 * ********************* Destructor *****************************
	 */

	template <class MaskPixelType,unsigned int MaskDimension, class CoordType >
		MaskImage<MaskPixelType,MaskDimension, CoordType>
		::~MaskImage()
	{		
	} // end Destructor
	

	/**
	 * ********************* Erode **********************************
	 */

	template <class MaskPixelType, unsigned int MaskDimension, class CoordType >
		typename MaskImage<MaskPixelType, MaskDimension, CoordType>::Pointer
		MaskImage<MaskPixelType, MaskDimension, CoordType>
		::Erode( const RadiusValueType radius )
	{
		/** Make Erode-filters.*/
		typename ErodeFilter::Pointer erosion[ ImageDimension ];
		for ( unsigned int i = 0; i < MaskDimension; i++ )
		{
			erosion[ i ] = ErodeFilter::New();
		}
		
		/** Declare radiusarray.*/
		RadiusType radiusarray;
		
		/** Set as input ..... */
		erosion[ 0 ]->SetInput(this);
		
		/**  */
		for ( unsigned int i = 0; i < MaskDimension; i++ )
		{			
			radiusarray.Fill(0);
			radiusarray.SetElement( i, radius );
			
			this->m_Ball.SetRadius( radiusarray );
			this->m_Ball.CreateStructuringElement();
			
			erosion[ i ]->SetKernel( this->m_Ball );

			if ( i > 0 ) erosion[ i ]->SetInput( erosion[ i - 1 ]->GetOutput() );			
		}
		
		/***/
		Self::Pointer dummyoutput = Self::New();
		dummyoutput->SetRegions( this->GetLargestPossibleRegion() );
		dummyoutput->SetOrigin( this->GetOrigin() );
		dummyoutput->SetSpacing( this->GetSpacing() );
		dummyoutput->Allocate();
		erosion[ MaskDimension - 1 ]->GraftOutput( dummyoutput );

		/** Do the erosion. */
		try
		{
			erosion[ MaskDimension - 1 ]->Update();
		}
		catch( itk::ExceptionObject & excp )
		{
			/** Add information to the exception. */
			excp.SetLocation( "MaskImage - Erode()" );
			std::string err_str = excp.GetDescription();
			err_str += "\nError while eroding the mask.\n";
			excp.SetDescription( err_str );
			/** Pass the exception to an higher level. */
			throw excp;
		}
		
		/** return a value.*/
		return dummyoutput;
		
	} // end Erode
	

	/**
	 * ********************* Dilate *********************************
	 */

	template <class MaskPixelType, unsigned int MaskDimension, class CoordType >
		typename MaskImage<MaskPixelType, MaskDimension, CoordType>::Pointer
		MaskImage<MaskPixelType, MaskDimension, CoordType>
		::Dilate( const RadiusValueType radius )
	{
		/** Make Dilate-filters.*/
		typename DilateFilter::Pointer	dilation[ ImageDimension ];
		for ( unsigned int i = 0; i < MaskDimension; i++ )
		{
			dilation[ i ] = DilateFilter::New();
		}
		
		/***/
		RadiusType radiusarray;
		
		/***/
		dilation[ 0 ]->SetInput(this);
		
		/***/
		for ( unsigned int i = 0; i < MaskDimension; i++ )
		{			
			radiusarray.Fill(0);
			radiusarray.SetElement( i, radius );
			
			this->m_Ball.SetRadius( radiusarray );
			this->m_Ball.CreateStructuringElement();
			
			dilation[ i ]->SetKernel( this->m_Ball );

			if ( i > 0 ) dilation[ i ]->SetInput( dilation[ i - 1 ]->GetOutput() );			
		}
		
		/***/
		Self::Pointer dummyoutput = Self::New();
		dummyoutput->SetRegions( this->GetLargestPossibleRegion() );
		dummyoutput->SetOrigin( this->GetOrigin() );
		dummyoutput->SetSpacing( this->GetSpacing() );
		dummyoutput->Allocate();
		dilation[ MaskDimension - 1 ]->GraftOutput( dummyoutput );
		
		/** Do the dilation. */
		try
		{
			dilation[ MaskDimension - 1 ]->Update();
		}
		catch( itk::ExceptionObject & excp )
		{
			/** Add information to the exception. */
			excp.SetLocation( "MaskImage - Dilate()" );
			std::string err_str = excp.GetDescription();
			err_str += "\nError while dilating the mask.\n";
			excp.SetDescription( err_str );
			/** Pass the exception to an higher level. */
			throw excp;
		}
		
		/** return a value.*/
		return dummyoutput;
		
	} // end Dilate
	
	
} // end namespace itk


#endif // end #ifndef _itkMaskImage_cxx

