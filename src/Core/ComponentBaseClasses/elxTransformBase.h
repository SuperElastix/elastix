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
	 * \class TransformBase
	 * \brief This class is the base for all Transforms
	 *
	 * This class contains the common functionality for all Transforms.
	 * Everything that is the same for every transform should be in this class.
	 *
	 * \ingroup Transforms
	 * \ingroup ComponentBaseClasses
	 */

	template <class TElastix>
		class TransformBase : public BaseComponentSE<TElastix>
	{
	public:

		/** Standard itk. */
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
		
		/** Get	the dimension of the fixed image. */
		itkStaticConstMacro( FixedImageDimension, unsigned int, FixedImageType::ImageDimension );
		/** Get	the dimension of the moving image. */
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

		/** Cast to ITKBaseType. */
		virtual ITKBaseType * GetAsITKBaseType(void)
		{
			return dynamic_cast<ITKBaseType *>(this);
		}

		/** Execute stuff before everything else:
		 * \li Check the appearance of an initial transform.
		 */
		virtual int BeforeAllBase(void);
		/** Execute stuff before the actual transformation:
		 * \li Check the appearance of inputpoints to be transformed.
		 */
		virtual int BeforeAllTransformix(void);
		/** Execute stuff before the actual registration:
		 * \li stuff regarding the initial tranform.
		 */
		virtual void BeforeRegistrationBase(void);
		/** Execute stuff after the actual registration:
		 * \li Get and set the final parameters for the resampler.
		 */
		virtual void AfterRegistrationBase(void);

		/** Get the initial transform. */
		virtual ObjectType * GetInitialTransform(void);
		/** Set the initial transform. */
		virtual void SetInitialTransform( ObjectType * _arg );

		/** Set the TransformParametersFileName. */
		virtual void SetTransformParametersFileName( const char * filename );
		/** Get the TransformParametersFileName. */
		itkGetStringMacro( TransformParametersFileName );

		/** Function to read transform-parameters from a file. */
		virtual void ReadFromFile(void);
		/** Function to write transform-parameters to a file. */
		virtual void WriteToFile( const ParametersType & param );
		/** Function to write transform-parameters to a file. */
		virtual void WriteToFile(void);
		/** Function to read the initial transform parameters from a file. */
		virtual void ReadInitialTransformFromFile(
			const char * transformParameterFileName);

		/** Function to transform coordinates from fixed to moving image. */
		virtual void TransformPoints(void);
		/** Function to transform coordinates from fixed to moving image. */
		virtual void TransformPointsSomePoints( std::string filename );
		/** Function to transform all coordinates from fixed to moving image. */
		virtual void TransformPointsAllPoints(void);		

	protected:

		/** The constructor. */
		TransformBase();
		/** The destructor. */
		virtual ~TransformBase();

		/** Member variables.*/
		ParametersType *			m_TransformParametersPointer;
		ConfigurationPointer	m_ConfigurationInitialTransform;
		std::string						m_TransformParametersFileName;

	private:

		/** The private constructor. */
		TransformBase( const Self& );		// purposely not implemented
		void operator=( const Self& );	// purposely not implemented

	}; // end class TransformBase


} // end namespace elastix


#ifndef ITK_MANUAL_INSTANTIATION
#include "elxTransformBase.hxx"
#endif

#endif // end #ifndef __elxTransformBase_h
