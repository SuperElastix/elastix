#ifndef __elxTransformBase_h
#define __elxTransformBase_h

/** Needed for the macros */
#include "elxMacro.h"

#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageFileWriter.h"
#include "itkVector.h"
#include "elxBaseComponentSE.h"
#include "itkTransform.h"
#include "itkTransformGrouperInterface.h"
#include "elxComponentDatabase.h"

/** Needed by most transforms: */
#include "itkTransformGrouper.h"

//#include <fstream>
#include <fstream>
#include <iomanip>

namespace elastix
{
  using namespace itk;


	/**
	 * ********************** TransformBase *************************
	 *
	 * This class
	 */

	template <class TElastix>
		class TransformBase : public BaseComponentSE<TElastix>
	{
	public:

		/** Standard.*/
		typedef TransformBase								Self;
		typedef BaseComponentSE<TElastix>		Superclass;

		/** Typedef's from Superclass.*/
		typedef typename Superclass::ElastixType						ElastixType;
		typedef typename Superclass::ElastixPointer					ElastixPointer;
		typedef typename Superclass::ConfigurationType			ConfigurationType;
		typedef typename Superclass::ConfigurationPointer		ConfigurationPointer;
		typedef typename ConfigurationType::ArgumentMapType	ArgumentMapType;
		typedef typename ArgumentMapType::value_type				ArgumentMapEntryType;
		typedef typename Superclass::RegistrationType				RegistrationType;
		typedef typename Superclass::RegistrationPointer		RegistrationPointer;

		/** Elastix typedef's.*/
		typedef typename ElastixType::CoordRepType								CoordRepType;		
		typedef typename ElastixType::FixedInternalImageType			FixedImageType;
		typedef typename ElastixType::MovingInternalImageType			MovingImageType;
		typedef typename FixedImageType::SizeType									SizeType;
		typedef typename FixedImageType::IndexType								IndexType;
		typedef typename FixedImageType::SpacingType							SpacingType;
		typedef typename FixedImageType::PointType								OriginType;
		//typedef typename FixedImageType::OffsetType								OffsetType;

		/** Typedef's from ComponentDatabase.*/
		typedef ComponentDatabase																	ComponentDatabaseType;
		typedef ComponentDatabaseType::ComponentDescriptionType		ComponentDescriptionType;
		typedef ComponentDatabase::PtrToCreator										PtrToCreator;
		
		/** Get Dimensions.*/
		itkStaticConstMacro( FixedImageDimension, unsigned int, FixedImageType::ImageDimension );
		itkStaticConstMacro( MovingImageDimension, unsigned int, MovingImageType::ImageDimension );

		/** Other typedef's.*/
		typedef itk::Object							ObjectType;
		typedef itk::Transform<
			CoordRepType,
			itkGetStaticConstMacro( FixedImageDimension ),
			itkGetStaticConstMacro( MovingImageDimension ) >					ITKBaseType;

		/** Typedef's from Transform.*/
		typedef typename ITKBaseType::ParametersType		ParametersType;
		typedef	typename ParametersType::ValueType			ValueType;

		/** Typedef's for TransformPoint.*/
		typedef typename ITKBaseType::InputPointType				InputPointType;
		typedef typename ITKBaseType::OutputPointType				OutputPointType;		
		typedef	Image< short,
			itkGetStaticConstMacro( FixedImageDimension ) >		DummyImageType;
		typedef typename DummyImageType::RegionType					RegionType;
		typedef Vector<
			float,
			itkGetStaticConstMacro( FixedImageDimension ) >		VectorType;
		typedef Image<
			VectorType,
			itkGetStaticConstMacro( FixedImageDimension ) >		OutputImageType;
		typedef typename OutputImageType::Pointer						OutputImagePointer;
		typedef ImageRegionConstIteratorWithIndex<
			DummyImageType >																	DummyIteratorType;
		typedef ImageRegionIteratorWithIndex<
			OutputImageType >																	OutputImageIteratorType;
		typedef ImageFileWriter< OutputImageType >					OutputFileWriterType;

		/** Cast to ITKBaseType.*/
		virtual ITKBaseType * GetAsITKBaseType(void)
		{
			return dynamic_cast<ITKBaseType *>(this);
		}

		/** Callback Methods.*/
		virtual int BeforeAllBase(void);
		virtual int BeforeAllTransformix(void);
		virtual void BeforeRegistrationBase(void);
		virtual void AfterRegistrationBase(void);

		/** Get/Set InitialTransform.*/
		virtual ObjectType * GetInitialTransform(void);
		virtual void SetInitialTransform( ObjectType * _arg );

		/** Get and Set the TransformParametersFileName.*/
		virtual void SetTransformParametersFileName( const char * filename );
		itkGetStringMacro( TransformParametersFileName );

		/** Function to read/write transform-parameters from/to a file.*/
		virtual void ReadFromFile(void);
		virtual void WriteToFile( const ParametersType & param );
		virtual void WriteToFile(void);
		virtual void ReadInitialTransformFromFile(
			const char * transformParameterFileName);

		/** Function to transform coordinates from fixed to moving image.*/
		virtual void TransformPoints(void);
		virtual void TransformPointsSomePoints( std::string filename );
		virtual void TransformPointsAllPoints(void);		

	protected:

		TransformBase();
		virtual ~TransformBase();

		/** Member variables.*/
		ParametersType *			m_TransformParametersPointer;
		ConfigurationPointer	m_ConfigurationInitialTransform;
		std::string						m_TransformParametersFileName;

	private:

		TransformBase( const Self& );		// purposely not implemented
		void operator=( const Self& );	// purposely not implemented

	}; // end class TransformBase


} // end namespace elastix


#ifndef ITK_MANUAL_INSTANTIATION
#include "elxTransformBase.hxx"
#endif

#endif // end #ifndef __elxTransformBase_h
