#ifndef __itkTransformGrouperInterface_h
#define __itkTransformGrouperInterface_h

#include "itkObject.h"
#include <string>

namespace itk
{
	
	/**
	 * ***************** TransformGrouperInterface ******************
	 *
	 * This class
	 */

	class TransformGrouperInterface
	{		
	public:
		
		/** Standard*/
		typedef TransformGrouperInterface Self;

		/** Other typedef's.*/
		typedef itk::Object ObjectType;
		typedef std::string GrouperDescriptionType;
		
		/** declare here already to allow elastix to use it;
		 * in the TransformGrouper an implementation is defined.
		 */		
		virtual ObjectType * GetInitialTransform(void) = 0;
		virtual void SetInitialTransform( ObjectType * _arg ) = 0;
		
		virtual int SetGrouper( const GrouperDescriptionType & name ) = 0;
		virtual const GrouperDescriptionType & GetNameOfDesiredGrouper(void) const = 0;
		virtual const GrouperDescriptionType & GetNameOfCurrentGrouper(void) const = 0;
				
	protected:
		
		TransformGrouperInterface() {}
		virtual ~TransformGrouperInterface() {}
		
	private:
		
		TransformGrouperInterface( const Self& );	// purposely not implemented
		void operator=( const Self& );						// purposely not implemented
		
	}; // end class TransformGrouperInterface
	
	
} // end namespace itk


#endif // end #ifndef __itkTransformGrouperInterface_h

