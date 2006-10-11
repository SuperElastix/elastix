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
		this->m_DeformationVectorFieldTransform = 
			DeformationVectorFieldTransformType::New();
		this->SetCurrentTransform(
			this->m_DeformationVectorFieldTransform );

	} // end Constructor
	

	
	/**
	 * ************************* ReadFromFile ************************
	 */

	template <class TElastix>
	void DeformationFieldTransform<TElastix>::
		ReadFromFile(void)
	{
		// \todo Test this ReadFromFile function.

    /** Make sure that the Transformbase::WriteToFile() does
		 * not read the transformParameters in the file. */
		this->SetReadWriteTransformParameters( false );
    
		/** Call the ReadFromFile from the TransformBase. */
		this->Superclass2::ReadFromFile();

		/** Setup VectorImageReader. */
		typedef ImageFileReader< CoefficientVectorImageType >	VectorReaderType;
		typename VectorReaderType::Pointer vectorReader
			= VectorReaderType::New();

		/** Read deformationFieldImage-name from parameter-file. */
		std::string fileName = "";
		this->m_Configuration->ReadParameter( fileName,
			"DeformationFieldFileName", 0 );
    if ( fileName = "" )
    {
      xl::xout["error"] << "ERROR: the entry (DeformationFieldFileName \"...\") is missing in the transform parameter file!" << std::endl;
      itkExceptionMacro( << "Error while reading transform parameter file!" );
    }

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
		this->m_DeformationVectorFieldTransform->
			SetCoefficientVectorImage( vectorReader->GetOutput() );
		
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
		 * not write the transformParameters in the file. */
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
		this->m_Configuration->ReadParameter(	resultImageFormat, "ResultImageFormat", 0, true );
		std::ostringstream makeFileName( "" );
		makeFileName << this->m_Configuration->GetCommandLineArgument( "-out" )
			<< "DeformationFieldImage"
			<< lastpart
			<< "." << resultImageFormat;
		xout["transpar"] << "(DeformationFieldFileName \""
			<< makeFileName.str() << "\")" << std::endl;

		/** Create a deformationField image. */
		CoefficientVectorImagePointer deformationFieldImage;
		this->m_DeformationVectorFieldTransform->
			GetCoefficientVectorImage( deformationFieldImage );

		/** Write the deformation field image. */
		typedef itk::ImageFileWriter< CoefficientVectorImageType > VectorWriterType;
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

