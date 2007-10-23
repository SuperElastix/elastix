#ifndef __elxTransformBase_h
#define __elxTransformBase_h

/** Needed for the macros */
#include "elxMacro.h"


#include "elxBaseComponentSE.h"
#include "itkTransform.h"
#include "itkCombinationTransform.h"
#include "elxComponentDatabase.h"
#include "elxProgressCommand.h"

#include <fstream>
#include <iomanip>

namespace elastix
{
  //using namespace itk; //Not here, because a TransformBase class was recently added to ITK...

	/**
	 * \class TransformBase
	 * \brief This class is the base for all Transforms.
	 *
	 * This class contains the common functionality for all Transforms.
	 *
	 * The parameters used in this class are:
	 * \parameter HowToCombineTransforms: Indicates how to use the initial transform\n
	 *   (given by the command-line argument -t0, or, if using multiple parameter files,
	 *   by the result of registration using the previous parameter file). Possible options
	 *   are "Add" and "Compose".\n
   *   "Add" combines the initial transform \f$T_0\f$ and the current
	 *   transform \f$T_1\f$ (which is currently optimised) by
   *   addition: \f$T(x) = T_0(x) + T_1(x)\f$;\n
	 *   "Compose" by composition: \f$T(x) = T_1 ( T_0(x) )\f$.\n
	 *   example: <tt>(HowToCombineTransforms "Add")
	 *   Default: "Add".
	 *
	 * The command line arguments used by this class are:
	 * \commandlinearg -t0: optional argument for elastix for specifying an initial transform
	 *		parameter file. \n
	 *		example: <tt>-t0 TransformParameters.txt</tt> \n
	 *
	 * \ingroup Transforms
	 * \ingroup ComponentBaseClasses
	 */

	template <class TElastix>
		class TransformBase : public BaseComponentSE<TElastix>
	{
	public:

		/** Standard ITK stuff. */
		typedef TransformBase								Self;
		typedef BaseComponentSE<TElastix>		Superclass;

		/** Run-time type information (and related methods). */
		itkTypeMacro( TransformBase, BaseComponentSE );

		/** Typedef's from Superclass. */
		typedef typename Superclass::ElastixType						ElastixType;
		typedef typename Superclass::ElastixPointer					ElastixPointer;
		typedef typename Superclass::ConfigurationType			ConfigurationType;
		typedef typename Superclass::ConfigurationPointer		ConfigurationPointer;
		typedef typename ConfigurationType::ArgumentMapType	ArgumentMapType;
		typedef typename ArgumentMapType::value_type				ArgumentMapEntryType;
		typedef typename Superclass::RegistrationType				RegistrationType;
		typedef typename Superclass::RegistrationPointer		RegistrationPointer;

		/** Elastix typedef's. */
		typedef typename ElastixType::CoordRepType								CoordRepType;		
		typedef typename ElastixType::FixedImageType			        FixedImageType;
		typedef typename ElastixType::MovingImageType			        MovingImageType;
		
		/** Typedef's from ComponentDatabase. */
		typedef ComponentDatabase																	ComponentDatabaseType;
		typedef ComponentDatabaseType::ComponentDescriptionType		ComponentDescriptionType;
		typedef ComponentDatabase::PtrToCreator										PtrToCreator;

    /** Typedef for the ProgressCommand. */
    typedef elx::ProgressCommand          ProgressCommandType;
		
		/** Get	the dimension of the fixed image. */
		itkStaticConstMacro( FixedImageDimension, unsigned int, FixedImageType::ImageDimension );
		/** Get	the dimension of the moving image. */
		itkStaticConstMacro( MovingImageDimension, unsigned int, MovingImageType::ImageDimension );

		/** Other typedef's. */
		typedef itk::Object							                            ObjectType;
		typedef itk::Transform<
			CoordRepType,
			itkGetStaticConstMacro( FixedImageDimension ),
			itkGetStaticConstMacro( MovingImageDimension ) >					ITKBaseType;
		typedef itk::CombinationTransform<CoordRepType,
			itkGetStaticConstMacro( FixedImageDimension ) >						CombinationTransformType;	
		typedef typename 
			CombinationTransformType::InitialTransformType						InitialTransformType;

		/** Typedef's from Transform. */
		typedef typename ITKBaseType::ParametersType		ParametersType;
		typedef	typename ParametersType::ValueType			ValueType;

 		/** Typedef's for TransformPoint. */
		typedef typename ITKBaseType::InputPointType				InputPointType;
		typedef typename ITKBaseType::OutputPointType				OutputPointType;		
		
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
		 * \li Set the initial tranform and how to group transforms.
		 */
		virtual void BeforeRegistrationBase(void);

		/** Execute stuff after the registration:
		 * \li Get and set the final parameters for the resampler.
		 */
		virtual void AfterRegistrationBase(void);

		/** Get the initial transform. */
		virtual const InitialTransformType * GetInitialTransform(void) const;
		/** Set the initial transform. */
		virtual void SetInitialTransform( InitialTransformType * _arg );

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
		/** Macro for reading and writing the transform parameters in WriteToFile or not. */
		virtual void SetReadWriteTransformParameters( const bool _arg );
		/** Function to read the initial transform parameters from a file. */
		virtual void ReadInitialTransformFromFile(
			const char * transformParameterFileName);

		/** Function to transform coordinates from fixed to moving image. */
		virtual void TransformPoints(void);
		/** Function to transform coordinates from fixed to moving image. */
		virtual void TransformPointsSomePoints( std::string filename );
		/** Function to transform all coordinates from fixed to moving image. */
		virtual void TransformPointsAllPoints(void);

		/** Makes sure that the final parameters from the registration components
		 * are copied, set, and stored.
		 */
		virtual void SetFinalParameters(void);

	protected:

		/** The constructor. */
		TransformBase();
		/** The destructor. */
		virtual ~TransformBase();

		/** Member variables.*/
		ParametersType *			m_TransformParametersPointer;
		ConfigurationPointer	m_ConfigurationInitialTransform;
		std::string						m_TransformParametersFileName;
		ParametersType				m_FinalParameters;

	private:

		/** The private constructor. */
		TransformBase( const Self& );		// purposely not implemented
		/** The private copy constructor. */
		void operator=( const Self& );	// purposely not implemented

		/** Boolean to decide whether or not the transform parameters are written. */
		bool		m_ReadWriteTransformParameters;

	}; // end class TransformBase


} // end namespace elastix


#ifndef ITK_MANUAL_INSTANTIATION
#include "elxTransformBase.hxx"
#endif

#endif // end #ifndef __elxTransformBase_h
