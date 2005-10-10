#ifndef __elxDeformationFieldTransform_HXX__
#define __elxDeformationFieldTransform_HXX__

#include "elxDeformationFieldTransform.h"

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

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
		/** Initialize. */

	} // end Constructor
	

	/**
	 * ******************* BeforeRegistration ***********************
	 *

	template <class TElastix>
		void DeformationFieldTransform<TElastix>
		::BeforeRegistration(void)
	{
		
	} // end BeforeRegistration
	

	/**
	 * ***************** BeforeEachResolution ***********************
	 *

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
		// \todo Test this ReadFromFile function.

		/** Setup VectorImageReader. */
		typedef ImageFileReader< VectorImageType >	VectorReaderType;
		typename VectorReaderType::Pointer vectorReader
			= VectorReaderType::New();

		/** Read deformationFieldImage-name from parameter-file. */
		std::string fileName = "";
		this->m_Configuration->ReadParameter( fileName,
			"DeformationFieldFileName", 0 );

		/** Read deformationFieldImage from file. */
		vectorReader->SetFileName( fileName.c_str() );
		try
		{
			vectorReader->Update();
		}
		catch( itk::ExceptionObject & excp )
		{
			/** Add information to the exception. */
			excp.SetLocation( "DeformationFieldTransform - ReadFromFile()" );
			std::string err_str = excp.GetDescription();
			err_str += "\nError occured while reading the deformationField image.\n";
			excp.SetDescription( err_str );
			/** Pass the exception to an higher level. */
			throw excp;
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
		this->m_Configuration->ReadParameter( fileName,
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
		this->m_Configuration->ReadParameter( howToCombineTransforms, "HowToCombineTransforms", 0, true );
		
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
		// \todo Finish and Test this WriteToFile function.

    /** Make sure that the Transformbase::WriteToFile() does
		 * not write the transformParameters in the file.
		 */
		this->SetReadWriteTransformParameters( false );

		/** Call the WriteToFile from the TransformBase. */
		this->Superclass2::WriteToFile( param );

		/** Add some DeformationFieldTransform specific lines. */
		xout["transpar"] << std::endl << "// DeformationFieldTransform specific" << std::endl;

		/** Get the last part of the filename of the transformParameter-file,
		 * which is going to be part of the filename of the deformationField image.
		 */
		std::string ctpfn = this->GetElastix()->GetCurrentTransformParameterFileName();
		std::basic_string<char>::size_type pos = ctpfn.rfind( "TransformParameters." );
		std::string lastpart = ctpfn.substr( pos + 19, ctpfn.size() - pos - 19 - 4 );

		/** Create the filename of the deformationField image. */
		std::string resultImageFormat = "mhd";
		this->m_Configuration->ReadParameter(	resultImageFormat, "ResultImageFormat", 0 );
		std::ostringstream makeFileName( "" );
		makeFileName << this->m_Configuration->GetCommandLineArgument( "-out" )
			<< "DeformationFieldImage"
			<< lastpart
			<< "." << resultImageFormat;
		xout["transpar"] << "(DeformationFieldFileName \""
			<< makeFileName.str() << "\")" << std::endl;

		/** Create a deformationField image. */
		VectorImagePointer deformationFieldImage;
		this->GetCoefficientVectorImage( deformationFieldImage );

		/** Write the deformation field image. */
		typedef itk::ImageFileWriter< VectorImageType > VectorWriterType;
		typename VectorWriterType::Pointer writer
			= VectorWriterType::New();
		writer->SetFileName( makeFileName.str().c_str() );
		writer->SetInput( deformationFieldImage );

		/** Do the writing. */
		try
		{
			writer->Update();
		}
		catch( itk::ExceptionObject & excp )
		{
			/** Add information to the exception. */
			excp.SetLocation( "DeformationFieldTransform - WriteToFile()" );
			std::string err_str = excp.GetDescription();
			err_str += "\nError while writing the deformationFieldImage.\n";
			excp.SetDescription( err_str );
			/** Print the exception. */
			xl::xout["error"] << excp << std::endl;
		}

	} // end WriteToFile

	
} // end namespace elastix


#endif // end #ifndef __elxDeformationFieldTransform_HXX__

