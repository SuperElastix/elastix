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
		m_AffineTransform = AffineTransformType::New();
		this->SetCurrentTransform( this->m_AffineTransform );
	} // end Constructor
	
	
	/**
	 * ******************* BeforeRegistration ***********************
	 */
	
	template <class TElastix>
		void AffineTransformElastix<TElastix>
		::BeforeRegistration(void)
	{
		/** Task 1 - Set initial parameters. */
		this->InitializeTransform();
		
		
		/** Task 2 - Set the scales. */
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
	 * ************************* ReadFromFile ************************
	 */

	template <class TElastix>
	void AffineTransformElastix<TElastix>::
		ReadFromFile(void)
	{
		
		InputPointType centerOfRotationPoint;
		centerOfRotationPoint.Fill(0.0);
		bool pointRead = false;
		bool indexRead = false;

		/** Try first to read the CenterOfRotationPoint from the 
		 * transform parameter file, this is the new, and preferred
		 * way, since elastix 3.402.
		 */		 
		pointRead = ReadCenterOfRotationPoint(centerOfRotationPoint);

		/** If this did not succeed, probably a transform parameter file
		 * is trying to be read that was generated using an older elastix
		 * version. Try to read it as an index, and convert to point.
		 */
		if (!pointRead)
		{
      indexRead = ReadCenterOfRotationIndex(centerOfRotationPoint);
		}

		if (!pointRead && !indexRead)
		{
			xl::xout["error"] << "ERROR: No center of rotation is specified in the transform parameter file" << std::endl;
			itkExceptionMacro(<< "Transform parameter file is corrupt.")
		}

		/** Set the center in this Transform.*/
		this->m_AffineTransform->SetCenter( centerOfRotationPoint );

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
		/** Call the WriteToFile from the TransformBase.*/
		this->Superclass2::WriteToFile( param );

		/** Write AffineTransform specific things.*/
		xout["transpar"] << std::endl << "// AffineTransform specific" << std::endl;

		/** Set the precision of cout to 10. */
		xout["transpar"] << std::setprecision(10);

		/** Get the center of rotation point and write it to file */
		InputPointType rotationPoint = this->m_AffineTransform->GetCenter();
		xout["transpar"] << "(CenterOfRotationPoint ";
		for ( unsigned int i = 0; i < SpaceDimension - 1; i++ )
		{
			xout["transpar"] << rotationPoint[ i ] << " ";
		}
		xout["transpar"] << rotationPoint[ SpaceDimension - 1 ] << ")" << std::endl;

		/** Set the precision back to default value.*/
		xout["transpar"] << std::setprecision( this->m_Elastix->GetDefaultOutputPrecision() );

	
	} // end WriteToFile

  
	/**
	 * ************************* InitializeTransform *********************
	 */
	
	template <class TElastix>
		void AffineTransformElastix<TElastix>
		::InitializeTransform( void )
	{
		/** Set all parameters to zero (no rotations, no translation */
		this->m_AffineTransform->SetIdentity();
		
		/** Try to read CenterOfRotationIndex from parameter file,
		 * which is the rotationPoint, expressed in index-values.
		 */
    IndexType centerOfRotationIndex;
		bool CORInImage = true;
		bool centerGiven = true;
		SizeType fixedImageSize = this->m_Registration->GetAsITKBaseType()->
			GetFixedImage()->GetLargestPossibleRegion().GetSize();
		for ( unsigned int i = 0; i < SpaceDimension; i++ )
		{
			centerOfRotationIndex[ i ] = 0;
			/** Returns zero when parameter was in the parameter file */
			int returncode = this->m_Configuration->ReadParameter(
				centerOfRotationIndex[ i ], "CenterOfRotation", i, true );
			if ( returncode != 0 )
			{
				centerGiven &= false;
			}
			/** Check if CenterOfRotation has index-values within image.*/
			if ( centerOfRotationIndex[ i ] < 0 ||
				centerOfRotationIndex[ i ] > fixedImageSize[ i ] )
			{
				CORInImage = false;
			}
		}
		
		/** Give a warning if necessary.*/
		if ( !CORInImage && centerGiven )
		{
			xl::xout["warning"] << "WARNING: Center of Rotation is not within image boundaries!" << std::endl;
		}

		/** Check if user wants automatic transform initialization; false by default.
		 * If an initial transform is given, automatic transform initialization is 
		 * not possible */
		std::string automaticTransformInitializationString("false");
		bool automaticTransformInitialization = false;
		this->m_Configuration->ReadParameter(
			automaticTransformInitializationString,
			"AutomaticTransformInitialization", 0);
		if ( (automaticTransformInitializationString == "true") &&
			(this->Superclass1::GetInitialTransform() == 0) )
		{
			automaticTransformInitialization = true;
		}

		/** 
		 * Run the itkTransformInitializer if:
		 * - No center of rotation was given, or
		 * - The user asked for AutomaticTransformInitialization
		 */
		if ( !centerGiven || automaticTransformInitialization ) 
		{
	   
			/** Use the TransformInitializer to determine a center of 
			* of rotation and an initial translation */
			TransformInitializerPointer transformInitializer = 
				TransformInitializerType::New();
			transformInitializer->SetFixedImage(
				this->m_Registration->GetAsITKBaseType()->GetFixedImage() );
			transformInitializer->SetMovingImage(
				this->m_Registration->GetAsITKBaseType()->GetMovingImage() );
			transformInitializer->SetTransform(this->m_AffineTransform);
			transformInitializer->GeometryOn();
			transformInitializer->InitializeTransform();
		}

		/** Set the translation to zero, if no AutomaticTransformInitialization
		 * was desired 
		 */
    if ( !automaticTransformInitialization )
		{
			OutputVectorType noTranslation;
			noTranslation.Fill(0.0);
			this->m_AffineTransform->SetTranslation(noTranslation);
		}

		/** Set the center of rotation if it was entered by the user */
		if ( centerGiven )
		{
			/** Convert from index-value to physical-point-value.*/
			InputPointType centerOfRotationPoint;
			this->m_Registration->GetAsITKBaseType()->GetFixedImage()->
				TransformIndexToPhysicalPoint( centerOfRotationIndex, centerOfRotationPoint );
			this->m_AffineTransform->SetCenter(centerOfRotationPoint);
		}

		/** Apply the initial transform to the center of rotation, if 
		 * composition is used to combine the initial transform with the
		 * the current (affine) transform. */
		if ( (this->GetUseComposition()) && (this->Superclass1::GetInitialTransform() != 0) )
		{
			InputPointType transformedCenterOfRotationPoint = 
				this->Superclass1::GetInitialTransform()->TransformPoint( 
				this->m_AffineTransform->GetCenter() );
			this->m_AffineTransform->SetCenter(
				transformedCenterOfRotationPoint );
		}

		/** Set the initial parameters in this->m_Registration.*/
		this->m_Registration->GetAsITKBaseType()->
			SetInitialTransformParameters( this->GetParameters() );


	} // end InitializeTransform
	

	/**
	 * ******************** ReadCenterOfRotationIndex *********************
	 */

	template <class TElastix>
	bool AffineTransformElastix<TElastix>::
		ReadCenterOfRotationIndex(InputPointType & rotationPoint)
	{

		/** Try to read CenterOfRotationIndex from the transform parameter
		 * file, which is the rotationPoint, expressed in index-values.
		 */
		IndexType centerOfRotationIndex;
		bool centerGivenAsIndex = true;
		for ( unsigned int i = 0; i < SpaceDimension; i++ )
		{
			centerOfRotationIndex[ i ] = 0;
			/** Returns zero when parameter was in the parameter file */
			int returncode = this->m_Configuration->ReadParameter(
				centerOfRotationIndex[ i ], "CenterOfRotation", i, true );
			if ( returncode != 0 )
			{
				centerGivenAsIndex &= false;
			}
		} //end for i
		if (!centerGivenAsIndex)
		{
			return false;
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
			/** Read size from the parameter file. Zero by default, which is illegal. */
			size[ i ] = 0; 
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
		bool illegalSize = false;
		for ( unsigned int i = 0; i < SpaceDimension; i++ )
		{
			if ( size[ i ] == 0 )
			{
				illegalSize = true;
			}
		}
		if (illegalSize)
		{
			xl::xout["error"] << "ERROR: One or more image sizes are 0!" << std::endl;
			return false;
		}
		
		/** Make a temporary image with the right region info,
		* so that the TransformIndexToPhysicalPoint-functions will be right.
		*/
		typedef FixedImageType DummyImageType; 
		typename DummyImageType::Pointer dummyImage = DummyImageType::New();
		RegionType region;
		region.SetIndex( index );
		region.SetSize( size );
		dummyImage->SetRegions( region );
		dummyImage->SetOrigin( origin );
		dummyImage->SetSpacing( spacing );

		/** Convert center of rotation from index-value to physical-point-value.*/
		dummyImage->TransformIndexToPhysicalPoint( 
			centerOfRotationIndex, rotationPoint );

		/** Succesfully read centerOfRotation as Index */
		return true;

	} //end ReadCenterOfRotationIndex


		/**
	 * ******************** ReadCenterOfRotationPoint *********************
	 */

	template <class TElastix>
	bool AffineTransformElastix<TElastix>::
		ReadCenterOfRotationPoint(InputPointType & rotationPoint)
	{

		/** Try to read CenterOfRotationPoint from the transform parameter
		 * file, which is the rotationPoint, expressed in world coordinates.
		 */
		InputPointType centerOfRotationPoint;
		bool centerGivenAsPoint = true;
		for ( unsigned int i = 0; i < SpaceDimension; i++ )
		{
			centerOfRotationPoint[ i ] = 0;
			/** Returns zero when parameter was in the parameter file */
			int returncode = this->m_Configuration->ReadParameter(
				centerOfRotationPoint[ i ], "CenterOfRotationPoint", i, true );
			if ( returncode != 0 )
			{
				centerGivenAsPoint &= false;
			}
		} //end for i
		if (!centerGivenAsPoint)
		{
			return false;
		}
	
		/** copy the temporary variable into the output of this function,
		 * if everything went ok  */
		rotationPoint = centerOfRotationPoint;
		
		/** Succesfully read centerOfRotation as Point */
		return true;

	} //end ReadCenterOfRotationPoint


	
} // end namespace elastix


#endif // end #ifndef __elxAffineTransform_HXX_

