#ifndef __elxDeformationFieldTransform_HXX__
#define __elxDeformationFieldTransform_HXX__

#include "elxDeformationFieldTransform.h"
//#include "math.h"

//#include "itkImageFileReader.h"

namespace elastix
{
using namespace itk;


	/**
	 * ********************* Constructor ****************************
	 */
	
	template <class TElastix>
		DeformationFieldTransform<TElastix>
		::DeformationFieldTransform()
	{
		/** Initialize.*
		m_Coeffs1 = ImageType::New();
		m_Upsampler = UpsamplerType::New();
		m_Upsampler->SetSplineOrder(SplineOrder); 

		m_Caster	= TransformCastFilterType::New();
		m_Writer	= TransformWriterType::New();
		*/
	} // end Constructor
	

	/**
	 * ******************* BeforeRegistration ***********************
	 */

	template <class TElastix>
		void DeformationFieldTransform<TElastix>
		::BeforeRegistration(void)
	{
		
	} // end BeforeRegistration
	

	/**
	 * ***************** BeforeEachResolution ***********************
	 */

	template <class TElastix>
		void DeformationFieldTransform<TElastix>
		::BeforeEachResolution(void)
	{
	} // end BeforeEachResolution
	
	
	/**
	 * ************************* ReadFromFile ************************
	 */

	template <class TElastix>
	void DeformationFieldTransform<TElastix>::
		ReadFromFile(void)
	{
		/** Setup VectorImageReader. */
		typedef ImageFileReader< VectorImageType >	VectorReaderType;
		typename VectorReaderType::Pointer vectorReader
			= VectorReaderType::New();

		/** Read deformationFieldImage-name from parameter-file. */
		std::string fileName = "";
		m_Configuration->ReadParameter( fileName,
			"DeformationFieldFileName", 0 );

		/** Read deformationFieldImage from file. */
		vectorReader->SetFileName( fileName.c_str() );
		try
		{
			vectorReader->Update();
		}
		catch( itk::ExceptionObject & excp )
		{
			xl::xout["error"] << excp << std::endl;
		}		

		/** Set the deformationFieldImage in the
		 * itkDeformationVectorFieldTransform.
		 */
		this->Superclass1::SetCoefficientImage( vectorReader->GetOutput() );

		/** Do not call the ReadFromFile from the TransformBase,
		 * because that function tries to read parameters from the file,
		 * which is not necessary in this case, because the parameters are
		 * in the vectorImage.
		 * However, we have to copy the rest of the functionality:
		 */
		
		/** Get the InitialTransformName. */
		fileName = "";
		m_Configuration->ReadParameter( fileName,
			"InitialTransformParametersFileName", 0 );
		
		/** Call the function ReadInitialTransformFromFile.*/
		if ( fileName != "NoInitialTransform" )
		{			
			this->Superclass2::ReadInitialTransformFromFile( fileName.c_str() );
		} 

		/** Read from the configuration file how to combine the
		 * initial transform with the current transform.
		 */
		std::string howToCombineTransforms = "Add"; // default
		m_Configuration->ReadParameter( howToCombineTransforms, "HowToCombineTransforms", 0, true );
		
		/** Convert 'this' to a pointer to a TransformGrouperInterface and set how
		 * to combine the current transform with the initial transform */
		TransformGrouperInterface * thisAsGrouper = 
			dynamic_cast< TransformGrouperInterface * >(this);
		if ( thisAsGrouper )
		{
			thisAsGrouper->SetGrouper( howToCombineTransforms );
		}		

		/** Remember the name of the TransformParametersFileName.
		 * This will be needed when another transform will use this transform
		 * as an initial transform (see the WriteToFile method)
		 */
		this->Superclass2::SetTransformParametersFileName(
			this->Superclass2::GetConfiguration()->GetCommandLineArgument( "-tp" ) );

	} // end ReadFromFile


	/**
	 * ************************* WriteToFile ************************
	 *
	 * Saves the TransformParameters as a vector and if wanted
	 * also as a deformation field.
	 */

	template <class TElastix>
		void DeformationFieldTransform<TElastix>::
		WriteToFile( const ParametersType & param )
	{
		/** Call the WriteToFile from the TransformBase.*
		this->Superclass2::WriteToFile( param );

		/** Add some DeformationFieldTransform specific lines.*
		xout["transpar"] << std::endl << "// DeformationFieldTransform specific" << std::endl;

		/** Get the GridSize, GridIndex, GridSpacing and
		 * GridOrigin of this transform.
		 *
		SizeType size = this->GetGridRegion().GetSize();
		IndexType index = this->GetGridRegion().GetIndex();
		SpacingType spacing = this->GetGridSpacing();
		OriginType origin = this->GetGridOrigin();

		/** Write the GridSize of this transform.*
		xout["transpar"] << "(GridSize ";
		for ( unsigned int i = 0; i < SpaceDimension - 1; i++ )
		{
			xout["transpar"] << size[ i ] << " ";
		}
		xout["transpar"] << size[ SpaceDimension - 1 ] << ")" << std::endl;
		
		/** Write the GridIndex of this transform.*
		xout["transpar"] << "(GridIndex ";
		for ( unsigned int i = 0; i < SpaceDimension - 1; i++ )
		{
			xout["transpar"] << index[ i ] << " ";
		}
		xout["transpar"] << index[ SpaceDimension - 1 ] << ")" << std::endl;
		
		/** Set the precision of cout to 2, because GridSpacing and
		 * GridOrigin must have at least one digit precision.
		 *
		xout["transpar"] << std::setprecision(1);

		/** Write the GridSpacing of this transform.*
		xout["transpar"] << "(GridSpacing ";
		for ( unsigned int i = 0; i < SpaceDimension - 1; i++ )
		{
			xout["transpar"] << spacing[ i ] << " ";
		}
		xout["transpar"] << spacing[ SpaceDimension - 1 ] << ")" << std::endl;

		/** Write the GridOrigin of this transform.*
		xout["transpar"] << "(GridOrigin ";
		for ( unsigned int i = 0; i < SpaceDimension - 1; i++ )
		{
			xout["transpar"] << origin[ i ] << " ";
		}
		xout["transpar"] << origin[ SpaceDimension - 1 ] << ")" << std::endl;

		/** Set the precision back to default value.*
		xout["transpar"] << std::setprecision(6);

		/** If wanted, write the TransformParameters as deformation
		 * images to a file.
		 */

	} // end WriteToFile

	
} // end namespace elastix


#endif // end #ifndef __elxDeformationFieldTransform_HXX__

