#ifndef __elxAffineTransform_HXX_
#define __elxAffineTransform_HXX_

#include "elxAffineTransform.h"

namespace elastix
{
	using namespace itk;
	
	/**
	 * ********************* Constructor ****************************
	 */
	
	template <class TElastix>
		AffineTransformElastix<TElastix>
		::AffineTransformElastix()
	{
	} // end Constructor
	
	
	/**
	 * ******************* BeforeRegistration ***********************
	 */
	
	template <class TElastix>
		void AffineTransformElastix<TElastix>
		::BeforeRegistration(void)
	{
		/** Task 1 - Set initial parameters. */
		ParametersType dummyInitialParameters( this->GetNumberOfParameters() );
		dummyInitialParameters.Fill(0.0);
		unsigned int j = 0;

		/** Set it to the Identity-matrix. */
		for ( unsigned int i = 0; i < SpaceDimension; i++ )
		{
			dummyInitialParameters[ j ] = 1.0;
			j += SpaceDimension + 1;
		}

		/** And give it to this->m_Registration. */
		this->m_Registration->GetAsITKBaseType()
			->SetInitialTransformParameters( dummyInitialParameters );
		
		/** Task 2 - Set center of rotation. */
		InputPointType rotationPoint;
		this->CalculateRotationPoint( rotationPoint );
		this->Superclass1::SetCenter( rotationPoint );
		
		/** Task 3 - Set the scales. */
		/** Here is an heuristic rule for estimating good values for
		 * the rotation/translation scales.
		 *
		 * 1) Estimate the bounding box of your points (in physical units).
		 * 2) Take the 3D Diagonal of that bounding box
		 * 3) Multiply that by 10.0.
		 * 4) use 1.0 /[ value from (3) ] as the translation scaling value.
		 * 5) use 1.0 as the rotation scaling value.
		 *
		 * With this operation you bring the translation units
		 * to the range of rotations (e.g. around -1 to 1).
		 * After that, all your registration parameters are
		 * in the relaxed range of -1:1. At that point you
		 * can start setting your optimizer with step lengths
		 * in the ranges of 0.001 if you are conservative, or
		 * in the range of 0.1 if you want to live dangerously.
		 * (0.1 radians is about 5.7 degrees).
		 * 
		 * This heuristic rule is based on the naive assumption
		 * that your registration may require translations as
		 * large as 1/10 of the diagonal of the bounding box.
		 */

		/** Create the new scales. */
		ScalesType newscales( this->GetNumberOfParameters() );
		newscales.Fill( 1.0 );
		double dummy = 100000.0;

		/** The first SpaceDimension * SpaceDimension number of parameters
		 * represent rotations (4 in 2D and 9 in 3D).
		 */
		unsigned int RotationPart = SpaceDimension * SpaceDimension;

		/** this->m_Configuration->ReadParameter() returns 0 if there is a value given
		 * in the parameter-file, and returns 1 if there is no value given in the
		 * parameter-file.
		 * Check which option is used:
		 * - Nothing given in the parameter-file: rotations are scaled by the default
		 *		value 100000.0
		 * - Only one scale given in the parameter-file: rotations are scaled by this
		 *		value.
		 * - All scales are given in the parameter-file: each parameter is assigned its
		 *		own scale.
		 */

		/** Check the return values of ReadParameter. */
		std::vector<int> returnvalues( this->GetNumberOfParameters(), 5 );
		for ( unsigned int i = 0; i < this->GetNumberOfParameters(); i++ )
		{
			returnvalues[ i ] = this->m_Configuration->ReadParameter( dummy, "Scales", i, true );
		}

		/** Check which of the above options is used. */
		if ( returnvalues[ 0 ] == 1 )
		{
			/** In this case the first option is used. */
			for ( unsigned int i = 0; i < RotationPart; i++ )
			{
				newscales[ i ] = 100000.0;
			}
		}
		else if ( returnvalues[ 0 ] == 0 && returnvalues[ 1 ] == 1 )
		{
			/** In this case the second option is used. */
			double scale = 100000.0;
			this->m_Configuration->ReadParameter( scale, "Scales", 0 );
			for ( unsigned int i = 0; i < RotationPart; i++ )
			{
				newscales[ i ] = scale;
			}
		}
		else if ( returnvalues[ 0 ] == 0 && returnvalues[ this->GetNumberOfParameters() - 1 ] == 0 )
		{
			/** In this case the third option is used. */
			for ( unsigned int i = 0; i < this->GetNumberOfParameters(); i++ )
			{
				this->m_Configuration->ReadParameter( newscales[ i ], "Scales", i );
			}
		}
		else
		{
			/** In this case an error is made in the parameter-file.
			 * An error is thrown, because using erroneous scales in the optimizer
			 * can give unpredictable results.
			 */
			itkExceptionMacro( << "ERROR: The Scales-option in the parameter-file has not been set properly." );
		}

		/** And set the scales into the optimizer. */
		this->m_Registration->GetAsITKBaseType()->GetOptimizer()->SetScales( newscales );
		
	} // end BeforeRegistration
	
	
	/**
	 * ***************** CalculateRotationPoint *********************
	 */
	
	template <class TElastix>
		void AffineTransformElastix<TElastix>
		::CalculateRotationPoint( InputPointType & rotationPoint )
	{
		/** Fill rotationPoint.
		 * The CenterOfRotation is set by default to the middle
		 * of the fixed image. If CenterOfRotation is specified
		 * in the Parameterfile, then the default value is overwritten.
		 */
		
		/** Get fixed Image size. */
		SizeType fixedSize = this->m_Registration->GetAsITKBaseType()->
			GetFixedImage()->GetLargestPossibleRegion().GetSize();
		
		/** Fill CenterOfRotationIndices,
		 * which is the rotationPoint, expressed in index-values.
		 */
		IndexType CenterOfRotationIndices;
		bool CORInImage = true;
		for ( unsigned int i = 0; i < SpaceDimension; i++ )
		{
			CenterOfRotationIndices[ i ] = static_cast<IndexValueType>( fixedSize[ i ] / 2 );
			this->m_Configuration->ReadParameter( CenterOfRotationIndices[ i ], "CenterOfRotation", i, true );
			/** Check if CenterOfRotation has index-values within image. */
			if ( CenterOfRotationIndices[ i ] < 0 || CenterOfRotationIndices[ i ] > fixedSize[ i ] )
			{
				CORInImage = false;
			}
		}
		
		/** Give a warning if necessary. */
		if ( !CORInImage )
		{
			xl::xout["warning"] << "WARNING: Center of Rotation is not within image boundaries!" << std::endl;
		}
		
		/** Convert from index-value to physical-point-value. */
		this->m_Registration->GetAsITKBaseType()->GetFixedImage()->
			TransformIndexToPhysicalPoint( CenterOfRotationIndices, rotationPoint );

	} // end CalculateRotationPoint
	
	
	/**
	 * ************************* ReadFromFile ************************
	 */

	template <class TElastix>
	void AffineTransformElastix<TElastix>::
		ReadFromFile(void)
	{
		/** Read the center of rotation. */
		IndexType rotationIndex;
		for ( unsigned int i = 0; i < SpaceDimension; i++ )
		{
			rotationIndex[ i ] = 0;
			this->m_Configuration->ReadParameter( rotationIndex[ i ], "CenterOfRotation", i, true );		
		}
		
		/** Get spacing, origin and size of the fixed image.
		 * We put this in a dummy image, so that we can correctly
		 * calculate the center of rotation in world coordinates.
		 */
		SpacingType		spacing;
		IndexType			index;
		PointType			origin;
		SizeType			size;
		for ( unsigned int i = 0; i < SpaceDimension; i++ )
		{
			/** No default size. Read size from the parameter file. */
			this->m_Configuration->ReadParameter(	size[ i ], "Size", i );

			/** Default index. Read index from the parameter file. */
			index[ i ] = 0;
			this->m_Configuration->ReadParameter(	index[ i ], "Index", i );

			/** Default spacing. Read spacing from the parameter file. */
			spacing[ i ] = 1.0;
			this->m_Configuration->ReadParameter(	spacing[ i ], "Spacing", i );

			/** Default origin. Read origin from the parameter file. */
			origin[ i ] = 0.0;
			this->m_Configuration->ReadParameter(	origin[ i ], "Origin", i );
		}

		/** Check for image size. */
		unsigned int sum = 0;
		for ( unsigned int i = 0; i < SpaceDimension; i++ )
		{
			if ( size[ i ] == 0 ) sum++;
		}
		if ( sum > 0 )
		{
			xl::xout["error"] << "ERROR: One or more image sizes are 0!" << std::endl;
			/** \todo quit program nicely. */
		}
		
		/** Make a temporary image with the right region info,
		* so that the TransformIndexToPhysicalPoint-functions will be right.
		*/
		typename DummyImageType::Pointer dummyImage = DummyImageType::New();
		RegionType region;
		region.SetIndex( index );
		region.SetSize( size );
		dummyImage->SetRegions( region );
		dummyImage->SetOrigin( origin );
		dummyImage->SetSpacing( spacing );

		/** Convert center of rotation from index-value to physical-point-value. */
		InputPointType rotationPoint;
		dummyImage->TransformIndexToPhysicalPoint( rotationIndex, rotationPoint );

		/** Set it in this Transform. */
		this->SetCenter( rotationPoint );

		/** Call the ReadFromFile from the TransformBase.
		 * BE AWARE: Only call Superclass2::ReadFromFile() after CenterOfRotation
		 * is set, because it is used in the SetParameters()-function of this transform.
		 */
		this->Superclass2::ReadFromFile();

	} // end ReadFromFile


	/**
	 * ************************* WriteToFile ************************
	 */
	
	template <class TElastix>
		void AffineTransformElastix<TElastix>
		::WriteToFile( const ParametersType & param )
	{
		/** Call the WriteToFile from the TransformBase. */
		this->Superclass2::WriteToFile( param );

		/** Write AffineTransform specific things. */
		xout["transpar"] << std::endl << "// AffineTransform specific" << std::endl;

		/** Get the center of rotation and convert it from 
		 * physical-point-value to index-value.
		 */
		IndexType rotationIndex;
		InputPointType rotationPoint = this->GetCenter();
		this->m_Registration->GetAsITKBaseType()->GetFixedImage()->
			TransformPhysicalPointToIndex( rotationPoint, rotationIndex );

		/** Write the center of rotation. */
		xout["transpar"] << "(CenterOfRotation ";
		for ( unsigned int i = 0; i < SpaceDimension - 1; i++ )
		{
			xout["transpar"] << rotationIndex[ i ] << " ";
		}
		xout["transpar"] << rotationIndex[ SpaceDimension - 1 ] << ")" << std::endl;
	
	} // end WriteToFile
	
	
} // end namespace elastix


#endif // end #ifndef __elxAffineTransform_HXX_

